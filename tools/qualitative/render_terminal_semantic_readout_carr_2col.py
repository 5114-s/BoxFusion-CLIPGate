#!/usr/bin/env python3
"""Render a verified 2x3 terminal semantic-readout figure.

The figure uses archived terminal predictions, the frozen-CLIP semantic cache,
the exact RGB view selected by the semantic protocol, and the real ScanNet
mesh.  It performs no detector or semantic-model inference.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import pickle
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/extra/ZhaoX/scannet_data/scans")
RECEIPTS = ROOT / (
    "reports/official_github_boxfusion_vs_final_recar_public_cli_"
    "coldcache_nowarmup100_20261009/receipts/recar"
)
SEMANTIC = ROOT / "reports/scannet_zero_shot_open_vocab_boxfusion_original_20261010"
OUT = ROOT / "figure_assets/open_vocab_readout_carr_2x2_20261106"

COLORS = {
    "ink": "#172A40",
    "muted": "#5D6A7B",
    "native": "#236FB6",
    "gt": "#14945E",
    "ptr": "#F07820",
    "var": "#B27D00",
    "rule": "#D7DEE8",
    "panel": "#F7F9FC",
}
CORNER_CODES = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)

# Both rows are selected by the joint requirement of a genuine native miss,
# a branch-specific geometric recovery, and a correct frozen-CLIP Top-1 readout.
CASES = (
    {
        "row": "a",
        "branch": "PTR",
        "branch_key": "ptr",
        "source_key": "m1p",
        "scene": "scene0426_03",
        "prediction_index": 25,
        "frame_id": 575,
        "query": "window",
        "view_crop": [455, 55, 1296, 900],
    },
    {
        "row": "b",
        "branch": "VAR",
        "branch_key": "var",
        "source_key": "m1a",
        "scene": "scene0591_01",
        "prediction_index": 55,
        "frame_id": 1250,
        "query": "trash can",
        "view_crop": [15, 355, 565, 925],
    },
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_renderer():
    path = ROOT / "tools/qualitative/render_real_3d_motivation.py"
    spec = importlib.util.spec_from_file_location("terminal_readout_mesh", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def alignment(scene: str) -> np.ndarray:
    path = RAW / scene / f"{scene}.txt"
    row = next(line for line in path.read_text().splitlines()
               if line.startswith("axisAlignment"))
    return np.asarray(row.split("=", 1)[1].split(), dtype=float).reshape(4, 4)


def read_rows(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)[0]


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def project_pose(points: np.ndarray, camera_to_world: np.ndarray,
                 intrinsic: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    camera = homogeneous @ np.linalg.inv(camera_to_world).T
    xyz = camera[:, :3]
    return np.column_stack([
        intrinsic[0, 0] * xyz[:, 0] / xyz[:, 2] + intrinsic[0, 2],
        intrinsic[1, 1] * xyz[:, 1] / xyz[:, 2] + intrinsic[1, 2],
        xyz[:, 2],
    ])


def iou_matrix(boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 8, 3)
    gt = np.asarray(gt, dtype=float).reshape(-1, 8, 3)
    lo, hi = boxes.min(1), boxes.max(1)
    gl, gh = gt.min(1), gt.max(1)
    inter = np.maximum(
        0, np.minimum(hi[:, None], gh) - np.maximum(lo[:, None], gl)
    ).prod(2)
    union = (
        np.prod(hi - lo, axis=1)[:, None]
        + np.prod(gh - gl, axis=1)
        - inter
    )
    return inter / np.maximum(union, 1e-12)


def source_for_index(index: int, counts: dict) -> str:
    native = int(counts["native"])
    plr = int(counts["m1p"])
    if index < native:
        return "native"
    if index < native + plr:
        return "m1p"
    return "m1a"


def load_case(case: dict, renderer) -> dict:
    scene = case["scene"]
    matrix = alignment(scene)
    full_path = RECEIPTS / scene / "recar" / f"{scene}_boxes.pkl"
    native_path = RECEIPTS / scene / "native" / f"{scene}_boxes.pkl"
    diagnostic_path = RECEIPTS / scene / "online_diagnostics" / f"{scene}.json"
    semantic_path = SEMANTIC / "recar/semantic_cache" / f"{scene}.npz"
    gt_path = SEMANTIC / "gt_cache" / f"{scene}.npz"
    rgb_path = RAW / scene / "color" / f"{case['frame_id']}.jpg"
    pose_path = RAW / scene / "pose" / f"{case['frame_id']}.txt"
    intrinsic_path = RAW / scene / "intrinsic/intrinsic_color.txt"

    full_rows = read_rows(full_path)
    native_rows = read_rows(native_path)
    full_raw = np.asarray([row[1] for row in full_rows], dtype=float).reshape(-1, 8, 3)
    native_raw = np.asarray([row[1] for row in native_rows], dtype=float).reshape(-1, 8, 3)
    assert np.array_equal(full_raw[:len(native_raw)], native_raw)
    full = transform(full_raw, matrix)
    native = transform(native_raw, matrix)
    recovery = full[int(case["prediction_index"])]

    diagnostics = json.loads(diagnostic_path.read_text())
    counts = diagnostics["state"]["terminal_counts"]
    assert len(full) == int(counts["total"])
    assert source_for_index(int(case["prediction_index"]), counts) == case["source_key"]

    vocabulary = json.loads((SEMANTIC / "source_manifest.json").read_text())["query_vocabulary"]
    with np.load(semantic_path) as value:
        labels = value["labels"].copy()
        cosine = value["max_cosine"].copy()
        view_ids = value["view_ids"].copy()
        rectangles = value["rectangles"].copy()
        crop_hashes = value["crop_hashes"].copy()
    index = int(case["prediction_index"])
    assert int(view_ids[index]) == int(case["frame_id"])
    top1 = vocabulary[int(labels[index])]
    assert top1 == case["query"]

    with np.load(gt_path) as value:
        gt = value["boxes"].copy()
        gt_classes = value["classes"].copy()
    overlap = iou_matrix(full, gt)
    gt_index = int(np.argmax(overlap[index]))
    target_gt = gt[gt_index]
    gt_query = vocabulary[int(gt_classes[gt_index])]
    assert gt_query == case["query"]
    native_overlap = iou_matrix(native, target_gt[None])[:, 0]

    pose_raw = np.loadtxt(pose_path)
    intrinsic = np.loadtxt(intrinsic_path)
    pose_aligned = matrix @ pose_raw
    projected_recovery = project_pose(recovery, pose_aligned, intrinsic)
    exact_rect = np.asarray(rectangles[index], dtype=float)
    measured_rect = np.r_[
        projected_recovery[:, :2].min(0), projected_recovery[:, :2].max(0)
    ]
    projection_error = float(np.max(np.abs(exact_rect - measured_rect)))
    assert projection_error < 1e-8

    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    height, width = rgb.shape[:2]
    assert [width, height] == [1296, 968]

    xyz, _, display_rgb, faces, mesh_info = renderer.load_mesh(scene, matrix, [])
    projected_mesh = project_pose(xyz, pose_aligned, intrinsic)
    visible_faces = faces[np.all(projected_mesh[faces, 2] > 0.05, axis=1)]
    background = renderer.rasterize(
        projected_mesh, display_rgb, visible_faces, width, height
    )

    return {
        **case,
        "full_path": full_path,
        "native_path": native_path,
        "diagnostic_path": diagnostic_path,
        "semantic_path": semantic_path,
        "gt_path": gt_path,
        "rgb_path": rgb_path,
        "pose_path": pose_path,
        "intrinsic_path": intrinsic_path,
        "matrix": matrix,
        "pose_aligned": pose_aligned,
        "intrinsic": intrinsic,
        "rgb": rgb,
        "background": background,
        "native": native,
        "recovery": recovery,
        "gt": target_gt,
        "gt_index": gt_index,
        "iou": float(overlap[index, gt_index]),
        "native_best_iou": float(native_overlap.max(initial=0.0)),
        "cosine": float(cosine[index]),
        "rank": 1,
        "query_count": len(vocabulary),
        "rect": exact_rect,
        "crop_hash": str(crop_hashes[index]),
        "top1": top1,
        "counts": counts,
        "projection_error": projection_error,
        "mesh_info": mesh_info,
        "renderer": renderer,
    }


def axis_frame(ax, color=COLORS["rule"], width=0.65):
    ax.add_patch(Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes, fill=False,
        edgecolor=color, linewidth=width, zorder=100, clip_on=False,
    ))


def draw_projected_box(ax, payload: dict, box: np.ndarray, color: str,
                       *, width: float, dashed: bool = False,
                       alpha: float = 1.0, zorder: int = 8):
    projected = project_pose(box, payload["pose_aligned"], payload["intrinsic"])
    if not np.all(projected[:, 2] > 0):
        return None
    edges = payload["renderer"].box_edges(box)
    collection = LineCollection(
        projected[:, :2][edges], colors=color, linewidths=width,
        linestyles="--" if dashed else "-", alpha=alpha, zorder=zorder,
    )
    collection.set_path_effects([
        pe.Stroke(linewidth=width + 0.9, foreground="white", alpha=0.62),
        pe.Normal(),
    ])
    ax.add_collection(collection)
    return projected[:, :2]


def intersects_crop(projected: np.ndarray, crop: list[float]) -> bool:
    lo, hi = projected[:, :2].min(0), projected[:, :2].max(0)
    x1, y1, x2, y2 = crop
    return not (hi[0] < x1 or lo[0] > x2 or hi[1] < y1 or lo[1] > y2)


def draw_scene_panel(ax, payload: dict, *, recovered: bool):
    ax.imshow(payload["background"], interpolation="antialiased")
    crop = payload["view_crop"]
    shown_native = []
    draw_projected_box(
        ax, payload, payload["gt"], COLORS["gt"], width=1.35,
        dashed=True, zorder=10,
    )
    if recovered:
        draw_projected_box(
            ax, payload, payload["recovery"], COLORS[payload["branch_key"]],
            width=1.75, zorder=12,
        )

    x1, y1, x2, y2 = crop
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    axis_frame(ax)

    payload["shown_native_indices"] = shown_native


def draw_input_cell(fig, spec, payload: dict):
    full_ax = fig.add_subplot(spec)

    full_ax.imshow(payload["rgb"])
    x1, y1, x2, y2 = payload["rect"]
    full_ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1, fill=False,
        edgecolor=COLORS[payload["branch_key"]], linewidth=1.45, zorder=5,
        path_effects=[pe.Stroke(linewidth=2.5, foreground="white", alpha=.78), pe.Normal()],
    ))
    full_ax.axis("off")
    axis_frame(full_ax)
    full_ax.text(
        .5, -.045, f"“{payload['query']}”", transform=full_ax.transAxes,
        ha="center", va="top", fontsize=8.2, fontweight="semibold",
        color=COLORS["ink"], clip_on=False,
    )


def export_individual_panel(payload: dict, recovered: bool, path: Path):
    fig = plt.figure(figsize=(4.2, 4.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    draw_scene_panel(ax, payload, recovered=recovered)
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)


def render(payloads: list[dict]):
    width = 183 / 25.4
    fig = plt.figure(figsize=(width, 4.35))
    outer = fig.add_gridspec(
        2, 2, left=.060, right=.990, bottom=.115, top=.905,
        width_ratios=[.96, 1.08], hspace=.16, wspace=.055,
    )

    for row, payload in enumerate(payloads):
        draw_input_cell(fig, outer[row, 0], payload)
        recar_ax = fig.add_subplot(outer[row, 1])
        draw_scene_panel(recar_ax, payload, recovered=True)

        box = recar_ax.get_position()
        fig.text(
            .011, (box.y0 + box.y1) / 2,
            f"({payload['row']})\n{payload['branch']}",
            ha="center", va="center", rotation=90,
            fontsize=7.2, fontweight="bold", color=COLORS["ink"],
        )

    columns = (
        "Text query & RGB observation",
        "CaRR-3D semantic readout",
    )
    for col, title in enumerate(columns):
        position = outer[0, col].get_position(fig)
        fig.text(
            position.x0 + position.width / 2, .955, title,
            ha="center", va="center", fontsize=7.7,
            fontweight="bold", color=COLORS["ink"],
        )

    handles = [
        Line2D([], [], color=COLORS["gt"], lw=1.5, ls="--", label="Selected GT"),
        Line2D([], [], color=COLORS["ptr"], lw=1.8, label="PTR recovery"),
        Line2D([], [], color=COLORS["var"], lw=1.8, label="VAR recovery"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(.5, .035),
        ncol=3, frameon=False, fontsize=6.25,
        handlelength=1.8, columnspacing=1.2,
    )
    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"open_vocab_semantic_readout_2x2.{suffix}"
        fig.savefig(path, dpi=400, facecolor="white")
        outputs.append(path)
    preview = OUT / "open_vocab_semantic_readout_2x2_preview.png"
    fig.savefig(preview, dpi=180, facecolor="white")
    plt.close(fig)
    return outputs, preview


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": .6,
    })
    renderer = load_renderer()
    payloads = [load_case(case, renderer) for case in CASES]
    outputs, preview = render(payloads)

    individuals = []
    for payload in payloads:
        scene = payload["scene"]
        rgb_path = OUT / f"{payload['branch_key']}_{scene}_selected_rgb.png"
        crop_path = OUT / f"{payload['branch_key']}_{scene}_clip_crop.png"
        ctrl_path = OUT / f"{payload['branch_key']}_{scene}_boxfusion_ctrl.png"
        recar_path = OUT / f"{payload['branch_key']}_{scene}_carr3d.png"
        Image.fromarray(payload["rgb"]).save(rgb_path)
        x1, y1, x2, y2 = payload["rect"]
        Image.fromarray(payload["rgb"]).crop((x1, y1, x2, y2)).save(crop_path)
        export_individual_panel(payload, False, ctrl_path)
        export_individual_panel(payload, True, recar_path)
        individuals.extend([rgb_path, crop_path, ctrl_path, recar_path])

    manifest_cases = []
    source_hashes = {}
    for payload in payloads:
        source_paths = (
            payload["full_path"], payload["native_path"], payload["diagnostic_path"],
            payload["semantic_path"], payload["gt_path"], payload["rgb_path"],
            payload["pose_path"], payload["intrinsic_path"],
        )
        for path in source_paths:
            source_hashes[str(path)] = digest(path)
        manifest_cases.append({
            "row": payload["row"],
            "branch": payload["branch"],
            "source_key": payload["source_key"],
            "scene": payload["scene"],
            "prediction_index": payload["prediction_index"],
            "gt_index_in_semantic_cache": payload["gt_index"],
            "query": payload["query"],
            "top1": payload["top1"],
            "rank": payload["rank"],
            "query_count": payload["query_count"],
            "cosine": payload["cosine"],
            "frame_id": payload["frame_id"],
            "clip_rectangle_xyxy": payload["rect"].tolist(),
            "crop_sha256": payload["crop_hash"],
            "native_best_iou": payload["native_best_iou"],
            "recovery_iou": payload["iou"],
            "semantic_rectangle_projection_max_error_pixels": payload["projection_error"],
            "terminal_counts": payload["counts"],
            "shown_native_indices": payload["shown_native_indices"],
            "view_crop_xyxy": payload["view_crop"],
            "camera": "exact selected RGB camera pose and color intrinsics",
            "mesh": payload["mesh_info"],
        })

    manifest = {
        "schema": "carr3d.terminal_semantic_readout_figure.v1",
        "title": "Terminal open-vocabulary semantic readout of recovered boxes",
        "semantic_protocol": (
            "Frozen CLIP ViT-H-14, fixed 60-query vocabulary, maximum valid "
            "projected view; semantic readout applied to archived terminal boxes"
        ),
        "cases": manifest_cases,
        "source_sha256": source_hashes,
        "render_script": str(Path(__file__)),
        "render_script_sha256": digest(Path(__file__)),
        "outputs": {path.name: digest(path) for path in [*outputs, preview, *individuals]},
        "model_inference_rerun": False,
        "prediction_coordinates_edited": False,
        "generative_images_used": False,
        "claim_boundary": "selected examples; not unseen-category generalization evidence",
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "caption_zh.txt").write_text(
        "图X｜CaRR-3D恢复记录的文本语义读出。给定文本查询，冻结CLIP在60类词表中"
        "正确读出PTR和VAR恢复的三维记录。"
        "绿色虚线表示GT，橙色和金色分别表示PTR与VAR恢复框。\n",
        encoding="utf-8",
    )
    (OUT / "README.txt").write_text(
        "open_vocab_semantic_readout_2x2.png/.pdf/.svg：论文主图。\n"
        "open_vocab_semantic_readout_2x2_preview.png：快速预览。\n"
        "*_selected_rgb.png与*_clip_crop.png：实际语义视图及裁图。\n"
        "*_boxfusion_ctrl.png与*_carr3d.png：相同真实相机的独立三维面板。\n"
        "source_manifest.json：输入哈希、预测索引、IoU、语义读出与投影一致性审计。\n"
        "本图未重跑模型，未修改预测坐标，也未使用生成式图像。\n",
        encoding="utf-8",
    )
    bundle = OUT.with_suffix(".zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, f"{OUT.name}/{path.relative_to(OUT)}")
    print(json.dumps({
        "output": str(OUT),
        "figure": str(outputs[0]),
        "preview": str(preview),
        "bundle": str(bundle),
        "cases": [
            {"scene": p["scene"], "query": p["query"], "cosine": p["cosine"],
             "native_best_iou": p["native_best_iou"], "recovery_iou": p["iou"]}
            for p in payloads
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
