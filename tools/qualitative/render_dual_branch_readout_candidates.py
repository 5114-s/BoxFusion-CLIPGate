#!/usr/bin/env python3
"""Render real same-scene PTR/VAR readout candidates with audited boxes."""
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
GT_ROOT = ROOT / "evaluation/data_util/scannet_train_detection_data"
SCENES = (
    "scene0598_01", "scene0629_00", "scene0030_01",
    "scene0406_01", "scene0435_03", "scene0518_00", "scene0599_01",
)
COLORS = {"native": "#236FB6", "ptr": "#F07820", "var": "#E0A000"}
CORNER_CODES = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wall_ids(scene: str) -> list[int]:
    path = RAW / scene / f"{scene}.aggregation.json"
    objects = json.loads(path.read_text())["segGroups"]
    return [int(obj["objectId"]) for obj in objects if obj["label"] == "wall"]


def find_pair(native, full, scores, gt, iou_matrix):
    native_iou = iou_matrix(native, gt)
    full_iou = iou_matrix(full, gt)
    native_n = len(native)
    missed = np.where(native_iou.max(0) <= .15)[0]
    ptr = [i for i in range(native_n, len(full)) if scores[i] >= .05]
    var = [i for i in range(native_n, len(full)) if scores[i] < .05]
    options = []
    for pg in missed:
        if not ptr:
            continue
        pi = max(ptr, key=lambda i: full_iou[i, pg])
        pv = float(full_iou[pi, pg])
        for vg in missed:
            if vg == pg or not var:
                continue
            vi = max(var, key=lambda i: full_iou[i, vg])
            vv = float(full_iou[vi, vg])
            if min(pv, vv) >= .55:
                options.append((min(pv, vv), pv + vv, pi, pg, pv, vi, vg, vv))
    if not options:
        raise RuntimeError("no distinct high-IoU PTR/VAR pair")
    return max(options)


def native_display_indices(native, gt, excluded, iou_matrix, maximum=4):
    overlaps = iou_matrix(native, gt)
    choices = []
    for gi in range(len(gt)):
        if gi in excluded:
            continue
        pi = int(np.argmax(overlaps[:, gi]))
        value = float(overlaps[pi, gi])
        if value >= .25:
            choices.append((value, pi, gi))
    chosen, used = [], set()
    for value, pi, gi in sorted(choices, reverse=True):
        if pi not in used:
            chosen.append((pi, gi, value))
            used.add(pi)
        if len(chosen) == maximum:
            break
    return chosen


def add_box(ax, renderer, box, projection_args, color, lw=3.4):
    projected = renderer.project(box, *projection_args)
    edges = renderer.box_edges(box)
    ax.add_collection(LineCollection(projected[:, :2][edges], colors=color,
                                     linewidths=lw, alpha=.97, zorder=10))


def main():
    output = ROOT / "figure_assets/online_detection_readout_dual_branch_candidates_20261107"
    output.mkdir(parents=True, exist_ok=True)
    common = load_module("box_render_common", ROOT / "tools/qualitative/render_boxfusion_ctrl_recar_3d.py")
    renderer = common.load_renderer()
    receipt = RUN / "receipts/recar"
    manifest = {"schema": "carr3d.dual_branch_readout_candidates.v1", "scenes": []}

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    for scene in SCENES:
        matrix = common.alignment(scene)
        gt_array = np.load(GT_ROOT / f"{scene}_bbox.npy")
        gt = gt_array[:, None, :3] + (CORNER_CODES[None] - .5) * gt_array[:, None, 3:6]
        native_path = receipt / scene / "native" / f"{scene}_boxes.pkl"
        full_path = receipt / scene / "recar" / f"{scene}_boxes.pkl"
        native_pred = common.read_predictions(native_path, matrix)
        full_pred = common.read_predictions(full_path, matrix)
        native, full, scores = native_pred["boxes"], full_pred["boxes"], full_pred["scores"]
        if not np.array_equal(native_pred["raw"], full_pred["raw"][:len(native)]):
            raise RuntimeError(f"native prefix mismatch: {scene}")
        pair = find_pair(native, full, scores, gt, common.iou_matrix)
        _, _, ptr_index, ptr_gt, ptr_iou, var_index, var_gt, var_iou = pair
        selected_native = native_display_indices(native, gt, {ptr_gt, var_gt}, common.iou_matrix)

        removed = wall_ids(scene)
        xyz, rgb, display_rgb, faces, mesh_info = renderer.load_mesh(scene, matrix, removed)
        camera = common.make_camera(renderer, scene, xyz, faces, gt)
        projection_args = (np.asarray(camera["position"]), np.asarray(camera["look_at"]),
                           camera["focal_pixels"], np.asarray(camera["principal_point_pixels"]))
        projected = renderer.project(xyz, *projection_args)
        width, height = camera["image_size"]
        background = renderer.rasterize(projected, display_rgb, faces, width, height)

        fig = plt.figure(figsize=(7.5, 6), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(background, interpolation="antialiased")
        for ni, _, _ in selected_native:
            add_box(ax, renderer, native[ni], projection_args, COLORS["native"], lw=3.0)
        add_box(ax, renderer, full[ptr_index], projection_args, COLORS["ptr"], lw=4.3)
        add_box(ax, renderer, full[var_index], projection_args, COLORS["var"], lw=4.3)
        valid = np.unique(faces)
        uv = projected[valid, :2]
        display_boxes = [full[ptr_index], full[var_index]] + [native[i] for i, _, _ in selected_native]
        buv = renderer.project(np.concatenate(display_boxes), *projection_args)[:, :2]
        low = np.minimum(uv.min(0), buv.min(0)) - 18
        high = np.maximum(uv.max(0), buv.max(0)) + 18
        ax.set_xlim(low[0], high[0]); ax.set_ylim(high[1], low[1])
        ax.set_aspect("equal"); ax.axis("off")
        for ext in ("png", "svg"):
            fig.savefig(output / f"{scene}_dual_readout.{ext}", dpi=400, transparent=True,
                        bbox_inches="tight", pad_inches=.01)
        fig.savefig(output / f"{scene}_dual_readout_white.png", dpi=400, facecolor="white",
                    bbox_inches="tight", pad_inches=.01)
        plt.close(fig)

        # An audit preview uses dashed GT boxes and IoU labels; it is not a paper asset.
        fig = plt.figure(figsize=(7.5, 6), dpi=160)
        ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(background)
        for ni, _, _ in selected_native:
            add_box(ax, renderer, native[ni], projection_args, COLORS["native"], lw=2.5)
        add_box(ax, renderer, full[ptr_index], projection_args, COLORS["ptr"], lw=4.0)
        add_box(ax, renderer, full[var_index], projection_args, COLORS["var"], lw=4.0)
        for box, color in ((gt[ptr_gt], COLORS["ptr"]), (gt[var_gt], COLORS["var"])):
            projected_box = renderer.project(box, *projection_args)
            edges = renderer.box_edges(box)
            ax.add_collection(LineCollection(projected_box[:, :2][edges], colors=color,
                                             linewidths=2.0, linestyles="dashed", alpha=.85, zorder=11))
        ax.text(.02, .98, f"{scene}  PTR IoU {ptr_iou:.3f}  VAR IoU {var_iou:.3f}",
                transform=ax.transAxes, va="top", fontsize=13, weight="bold",
                bbox={"facecolor": "white", "alpha": .88, "edgecolor": "none"})
        ax.set_xlim(low[0], high[0]); ax.set_ylim(high[1], low[1]); ax.set_aspect("equal"); ax.axis("off")
        fig.savefig(output / f"{scene}_audit.png", dpi=250, facecolor="white",
                    bbox_inches="tight", pad_inches=.01)
        plt.close(fig)

        record = {
            "scene": scene,
            "native_source": native_pred["path"], "native_sha256": native_pred["sha256"],
            "full_source": full_pred["path"], "full_sha256": full_pred["sha256"],
            "ptr": {"prediction_index": int(ptr_index), "gt_index": int(ptr_gt),
                    "aabb_iou": float(ptr_iou), "score": float(scores[ptr_index])},
            "var": {"prediction_index": int(var_index), "gt_index": int(var_gt),
                    "aabb_iou": float(var_iou), "score": float(scores[var_index])},
            "native_display": [{"prediction_index": int(i), "gt_index": int(g), "aabb_iou": float(v)}
                               for i, g, v in selected_native],
            "removed_wall_instance_ids": removed, "mesh": mesh_info, "camera": camera,
            "prediction_geometry_edited": False, "selection": "GT-assisted display selection",
        }
        manifest["scenes"].append(record)
        print(json.dumps({"scene": scene, "ptr_iou": ptr_iou, "var_iou": var_iou,
                          "native_boxes": len(selected_native)}), flush=True)

    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
