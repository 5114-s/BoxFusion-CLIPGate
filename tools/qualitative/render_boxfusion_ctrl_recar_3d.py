#!/usr/bin/env python3
"""Render real, fixed-view 3D comparisons from archived prediction files.

Exports a uniform Top-50 view, an explicitly GT-assisted selected-target view,
and an all-output audit view. No detector inference or prediction edits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import pickle
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/extra/ZhaoX/scannet_data/scans")
RUN = ROOT / "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009"
SCENES = (
    {"scene": "scene0426_03", "gt_indices": [0, 1, 2, 4, 9], "walls": [16, 18, 27]},
    {"scene": "scene0606_01", "gt_indices": [0, 2, 3, 4, 5], "walls": [1, 2]},
)
METHODS = ("Ground truth", "BoxFusion", "BoxFusion-Ctrl", "ReCaR-3D")
METHOD_SLUGS = {"Ground truth": "gt", "BoxFusion": "boxfusion",
                "BoxFusion-Ctrl": "boxfusion_ctrl", "ReCaR-3D": "recar3d"}
COLORS = {"gt": "#14945E", "native": "#236FB6", "plr": "#F07820", "calr": "#B27D00"}
CORNER_CODES = np.asarray(list(itertools.product((0, 1), repeat=3)), dtype=float)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_renderer():
    path = ROOT / "tools/qualitative/render_real_3d_motivation.py"
    spec = importlib.util.spec_from_file_location("real_mesh_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alignment(scene):
    path = RAW / scene / f"{scene}.txt"
    line = next(x for x in path.read_text().splitlines() if x.startswith("axisAlignment"))
    return np.asarray(line.split("=", 1)[1].split(), dtype=float).reshape(4, 4)


def read_predictions(path, matrix):
    with path.open("rb") as f:
        rows = pickle.load(f)[0]
    raw = np.asarray([r[1] for r in rows], dtype=float).reshape(-1, 8, 3)
    boxes = raw @ matrix[:3, :3].T + matrix[:3, 3]
    return {"path": str(path), "sha256": sha(path), "raw": raw, "boxes": boxes,
            "scores": np.asarray([r[2] for r in rows], dtype=float)}


def iou_matrix(boxes, gt):
    lo, hi = boxes.min(1), boxes.max(1)
    gl, gh = gt.min(1), gt.max(1)
    inter = np.maximum(0, np.minimum(hi[:, None], gh) - np.maximum(lo[:, None], gl)).prod(2)
    return inter / np.maximum(np.prod(hi-lo, axis=1)[:, None] + np.prod(gh-gl, axis=1) - inter, 1e-12)


def selected_matches(boxes, gt, target_indices):
    """GT-assisted visual selection, explicitly not an inference readout rule."""
    overlaps = iou_matrix(boxes, gt)
    candidates = [(float(overlaps[i, j]), i, j)
                  for i in range(len(boxes)) for j in target_indices
                  if overlaps[i, j] > 0.15]
    chosen, used_gt = [], set()
    for value, i, j in sorted(candidates, key=lambda x: (-x[0], x[1], x[2])):
        if i not in chosen and j not in used_gt:
            chosen.append(i)
            used_gt.add(j)
    return chosen


def make_camera(renderer, scene, xyz, faces, gt):
    width, height = 1500, 1200
    if scene == "scene0426_03":
        old = json.loads((ROOT / "figure_assets/real_3d_motivation_20261026/fixed_camera.json").read_text())
        ratio = width / old["image_size"][0]
        return {"position": old["position"], "look_at": old["look_at"],
                "focal_pixels": old["focal_pixels"] * ratio,
                "principal_point_pixels": (np.asarray(old["principal_point_pixels"])*ratio).tolist(),
                "image_size": [width, height], "source": "existing verified perspective; uniformly rescaled"}
    lo, hi = gt.min((0, 1)), gt.max((0, 1))
    look = (lo+hi)/2
    look[2] = 0.95
    az, el = np.deg2rad([-58., 33.])
    direction = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    position = look + 12*direction
    principal = np.array([width/2, height/2], dtype=float)
    focal = 2000.
    valid = np.unique(faces)
    uv = renderer.project(xyz[valid], position, look, focal, principal)[:, :2]
    lower, upper = uv.min(0), uv.max(0)
    focal *= min(width*.90/(upper[0]-lower[0]), height*.88/(upper[1]-lower[1]))
    uv = renderer.project(xyz[valid], position, look, focal, principal)[:, :2]
    principal += np.array([width/2, height/2]) - (uv.min(0)+uv.max(0))/2
    return {"position": position.tolist(), "look_at": look.tolist(), "focal_pixels": float(focal),
            "principal_point_pixels": principal.tolist(), "image_size": [width, height],
            "source": "fixed perspective fitted to cutaway mesh, shared by every method"}


def scene_payload(spec, renderer, output, *, raster_scale=1):
    scene = spec["scene"]
    matrix = alignment(scene)
    gt_path = ROOT / "evaluation/data_util/scannet_train_detection_data" / f"{scene}_bbox.npy"
    gt_array = np.load(gt_path)
    gt = gt_array[:, None, :3] + (CORNER_CODES[None]-.5)*gt_array[:, None, 3:6]
    receipt = RUN / "receipts"
    sources = {
        "BoxFusion": receipt / "original" / scene / "native" / f"{scene}_boxes.pkl",
        "BoxFusion-Ctrl": receipt / "recar" / scene / "native" / f"{scene}_boxes.pkl",
        "ReCaR-3D": receipt / "recar" / scene / "recar" / f"{scene}_boxes.pkl",
    }
    predictions = {name: read_predictions(path, matrix) for name, path in sources.items()}
    native_n = len(predictions["BoxFusion-Ctrl"]["boxes"])
    full = predictions["ReCaR-3D"]
    assert np.array_equal(full["raw"][:native_n], predictions["BoxFusion-Ctrl"]["raw"])
    for name, pred in predictions.items():
        if name == "ReCaR-3D":
            pred["branches"] = ["native" if i < native_n else ("calr" if score < .05 else "plr")
                                for i, score in enumerate(pred["scores"])]
        else:
            pred["branches"] = ["native"]*len(pred["boxes"])
        pred["selected"] = selected_matches(pred["boxes"], gt, spec["gt_indices"])
        pred["top50"] = np.argsort(-pred["scores"], kind="stable")[:50].tolist()
        pred["all"] = list(range(len(pred["boxes"])))
    xyz, rgb, display_rgb, faces, mesh_info = renderer.load_mesh(scene, matrix, spec["walls"])
    camera = make_camera(renderer, scene, xyz, faces, gt)
    if raster_scale != 1:
        camera["focal_pixels"] *= raster_scale
        camera["principal_point_pixels"] = (np.asarray(camera["principal_point_pixels"])*raster_scale).tolist()
        camera["image_size"] = [int(x*raster_scale) for x in camera["image_size"]]
    args = (np.asarray(camera["position"]), np.asarray(camera["look_at"]), camera["focal_pixels"],
            np.asarray(camera["principal_point_pixels"]))
    projected = renderer.project(xyz, *args)
    width, height = camera["image_size"]
    background = renderer.rasterize(projected, display_rgb, faces, width, height)
    # Use the union of scene, GT and selected examples for every method's crop.
    visible = np.unique(faces)
    uv = projected[visible, :2]
    visual_boxes = [gt]
    visual_boxes.extend(pred["boxes"][pred["selected"]] for pred in predictions.values())
    visual_uv = renderer.project(np.concatenate(visual_boxes).reshape(-1, 3), *args)[:, :2]
    low = np.minimum(uv.min(0), visual_uv.min(0))-18*raster_scale
    high = np.maximum(uv.max(0), visual_uv.max(0))+18*raster_scale
    camera["display_limits"] = [float(low[0]), float(high[0]), float(high[1]), float(low[1])]
    Image.fromarray(background).save(output / f"{scene}_mesh.png")
    arrays = {"gt_boxes": gt, "axis_alignment": matrix}
    for name, pred in predictions.items():
        prefix = {"BoxFusion": "boxfusion", "BoxFusion-Ctrl": "ctrl", "ReCaR-3D": "recar"}[name]
        arrays[prefix+"_boxes"] = pred["boxes"]
        arrays[prefix+"_scores"] = pred["scores"]
        arrays[prefix+"_selected_indices"] = np.asarray(pred["selected"], dtype=int)
        arrays[prefix+"_top50_indices"] = np.asarray(pred["top50"], dtype=int)
    np.savez_compressed(output / f"{scene}_measured_boxes.npz", **arrays)
    audit = {"scene": scene, "mesh": mesh_info, "camera": camera,
             "gt_path": str(gt_path), "gt_sha256": sha(gt_path), "gt_count": len(gt),
             "selected_gt_indices": spec["gt_indices"],
             "methods": {name: {"path": pred["path"], "sha256": pred["sha256"],
                                  "total_count": len(pred["boxes"]), "selected_indices": pred["selected"],
                                  "top50_indices": pred["top50"],
                                  "best_gt_aabb_iou": iou_matrix(pred["boxes"], gt).max(0).tolist(),
                                  "all_scores_tied": bool(len(np.unique(pred["scores"])) == 1)}
                         for name, pred in predictions.items()},
             "ctrl_geometry_equals_recar_native_prefix": True}
    return {"spec": spec, "gt": gt, "predictions": predictions, "background": background,
            "camera": camera, "projection_args": args, "audit": audit}


def draw_box(ax, box, renderer, projection_args, color, *, lw=.8, alpha=.92):
    projected = renderer.project(box, *projection_args)
    assert np.all(projected[:, 2] > 0)
    edges = renderer.box_edges(box)
    assert edges.shape == (12, 2)
    ax.add_collection(LineCollection(projected[:, :2][edges], colors=color, linewidths=lw,
                                     alpha=alpha, zorder=5))


def render_grid(payloads, renderer, output, mode, *, exports=("png", "pdf", "svg")):
    rows = len(payloads)
    fig, axes = plt.subplots(rows, 4, figsize=(14.4, 3.6*rows+.6), squeeze=False)
    fig.subplots_adjust(left=.024, right=.992, bottom=.17 if rows == 2 else .125,
                        top=.88 if rows == 2 else .92, wspace=.018, hspace=.26)
    for col, name in enumerate(METHODS):
        bounds = axes[0, col].get_position()
        fig.text(bounds.x0+bounds.width/2, .955 if rows == 2 else .970, name, fontsize=14, fontweight="bold",
                 ha="center", va="center", color="#172A40")
    for row, payload in enumerate(payloads):
        camera, scene = payload["camera"], payload["spec"]["scene"]
        for col, name in enumerate(METHODS):
            ax = axes[row, col]
            ax.imshow(payload["background"])
            if col == 0:
                gt_indices = payload["spec"]["gt_indices"] if mode == "selected" else range(len(payload["gt"]))
                for i in gt_indices:
                    draw_box(ax, payload["gt"][i], renderer, payload["projection_args"], COLORS["gt"], lw=1.02)
                small = f"{len(gt_indices)} GT objects"
            else:
                pred = payload["predictions"][name]
                for i in pred[mode]:
                    branch = pred["branches"][i]
                    draw_box(ax, pred["boxes"][i], renderer, payload["projection_args"], COLORS[branch],
                             lw=.8 if mode != "all" else .45, alpha=.93 if mode != "all" else .60)
                small = f"{len(pred[mode])} selected / {len(pred['boxes'])} total"
            ax.set_xlim(camera["display_limits"][:2])
            ax.set_ylim(camera["display_limits"][2:])
            ax.set_aspect("equal")
            ax.axis("off")
            ax.text(.5, -.045, small, transform=ax.transAxes, ha="center", va="top",
                    fontsize=8.3, color="#536273")
            if col == 0:
                bounds = ax.get_position()
                fig.text(.025, bounds.y1+.016, f"({chr(97+row)}) {scene}",
                         fontsize=10, fontweight="bold", color="#172A40")
    handles = [Line2D([0], [0], color=COLORS[k], lw=2, label=label) for k, label in (
        ("gt", "Ground truth"), ("native", "Native prediction"),
        ("plr", "PLR readout"), ("calr", "CALR readout"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .090 if rows == 2 else .060), ncol=4,
               frameon=False, fontsize=10.3, columnspacing=2.1)
    foot = {"selected": "Selected-target illustration (GT-assisted matching); omitted predictions include false positives.",
            "top50": "At most 50 predictions per scene by output score; all-tied BoxFusion uses archived row order.",
            "all": "All archived predictions shown, including duplicates and false positives; no post-render NMS."}[mode]
    fig.text(.5, .057 if rows == 2 else .039, foot, ha="center", fontsize=8.3, color="#536273")
    fig.text(.5, .032 if rows == 2 else .022, "Identical perspective and cutaway per scene. Scan mesh used only for visualization.",
             ha="center", fontsize=8.3, color="#536273")
    stem = output / f"comparison_3d_{mode}"
    for extension in exports:
        fig.savefig(stem.with_suffix("."+extension), dpi=400 if mode != "all" else 220,
                    facecolor="white")
    if mode != "all":
        fig.savefig(output / f"comparison_3d_{mode}_preview.png", dpi=150, facecolor="white")
    plt.close(fig)


def export_individuals(payload, renderer, output, mode):
    """Export each actual scene/method panel, without labels, at 3000x2400."""
    directory = output / f"individual_{mode}"
    directory.mkdir(parents=True, exist_ok=True)
    scene = payload["spec"]["scene"]
    camera = payload["camera"]
    records = []
    for name in METHODS:
        fig = plt.figure(figsize=(7.5, 6), dpi=400)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(payload["background"])
        if name == "Ground truth":
            indices = (payload["spec"]["gt_indices"] if mode == "selected"
                       else list(range(len(payload["gt"]))))
            for i in indices:
                draw_box(ax, payload["gt"][i], renderer, payload["projection_args"], COLORS["gt"], lw=1.25)
            total = len(payload["gt"])
        else:
            pred = payload["predictions"][name]
            indices = pred[mode]
            total = len(pred["boxes"])
            for i in indices:
                draw_box(ax, pred["boxes"][i], renderer, payload["projection_args"],
                         COLORS[pred["branches"][i]], lw=1.15)
        ax.set_xlim(camera["display_limits"][:2])
        ax.set_ylim(camera["display_limits"][2:])
        ax.set_aspect("equal")
        ax.axis("off")
        stem = directory / f"{scene}_{METHOD_SLUGS[name]}"
        extensions = ("png", "pdf", "svg") if mode == "selected" else ("png",)
        paths = []
        for extension in extensions:
            path = stem.with_suffix("."+extension)
            fig.savefig(path, dpi=400, transparent=True)
            paths.append(str(path.relative_to(output)))
        plt.close(fig)
        records.append({"scene": scene, "method": name, "mode": mode,
                        "selected_indices": [int(i) for i in indices], "total_records": total,
                        "files": paths, "png_size": [3000, 2400], "transparent": True})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "figure_assets/qualitative_3d_boxfusion_ctrl_recar_20261101")
    parser.add_argument("--include-third-scene", action="store_true")
    parser.add_argument("--export-individuals", action="store_true")
    parser.add_argument("--raster-scale", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    renderer = load_renderer()
    specs = list(SCENES)
    if args.include_third_scene:
        specs.append({"scene": "scene0432_01", "gt_indices": [0, 1, 2, 3, 4], "walls": [6, 8]})
    payloads = []
    individual_exports = []
    for spec in specs:
        payloads.append(scene_payload(spec, renderer, output, raster_scale=args.raster_scale))
        print(f"loaded and rasterized {spec['scene']}", flush=True)
        if args.export_individuals:
            for mode in ("selected", "top50"):
                individual_exports.extend(export_individuals(payloads[-1], renderer, output, mode))
            print(f"exported individual panels {spec['scene']}", flush=True)
    for mode in ("selected", "top50", "all"):
        render_grid(payloads, renderer, output, mode,
                    exports=("png",) if mode == "all" else ("png", "pdf", "svg"))
        print(f"exported {mode}", flush=True)
    manifest = {"schema": "recar3d.real_3d_method_comparison.v1", "source_run": str(RUN),
                "methods": list(METHODS), "scenes": [p["audit"] for p in payloads],
                "individual_exports": individual_exports, "scene_count": len(payloads),
                "prediction_coordinates_edited": False, "new_model_inference": False,
                "gpu_used": False, "generated_scene_content": False,
                "selected_view": "same selected GT targets; greedy one-to-one highest-IoU visual matches above 0.15; GT-assisted illustration, not detector output policy",
                "top50_view": "stable output-score Top-50 for every method; no extra score threshold or NMS",
                "all_view": "all terminal predictions, no removal",
                "box_lines": "12 measured cuboid edges; occluded edges also drawn as overlays",
                "view_scope": "shared scene viewport includes mesh, GT and selected-target examples; distant outlier predictions may project outside this fixed viewport",
                "calr_branch_identification": "records after identical native prefix with output score below 0.05",
                "source_run_note": "archived final three-factor HCSR/CALR-v1 efficiency-run outputs; not asserted identical to an independently rerun main AP table",
                "render_script_sha256": sha(Path(__file__)),
                "renderer_sha256": sha(ROOT / "tools/qualitative/render_real_3d_motivation.py")}
    (output / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    (output / "图注与使用说明.txt").write_text(
        "真实三维定性比较：GT / BoxFusion / BoxFusion-Ctrl / ReCaR-3D\n\n"
        "1. comparison_3d_top50：各方法按其输出分数逐场景最多显示50框，适合展示统一有限输出工作点。"
        "BoxFusion原始存档分数全部为1，采用原有行序稳定破同；不足50框的场景全部选取。"
        "图内注明选取数/总数。远离场景的异常框可能超出统一视野，视野外部分被裁剪。该图不会展示排名50之后的长尾恢复框。\n\n"
        "2. comparison_3d_selected：选定GT目标的定位/遗漏示例，按相同GT集合为各方法选择三维AABB IoU>0.15的代表预测，"
        "采用IoU降序一对一选取；明确使用了GT辅助选择。它不是完整预测或按分数筛选的输出，不能凭此推断误检率和整体精度。\n\n"
        "3. comparison_3d_all：全部原始输出审计图，包含重复框、误检和低分CALR长尾。\n\n"
        "Top-50版建议图注：ScanNet场景上的三维定性比较。从左至右为真实标注、BoxFusion、BoxFusion-Ctrl及ReCaR-3D。"
        "每种方法每场景选取按最终分数排序的前50个预测，选取数量与终态总数列于面板下方，超出统一视野的部分被裁剪。"
        "绿色为GT，蓝色为原生预测，橙色和金色分别为PLR与CALR读出。每行共享相同透视相机、网格、剖切和颜色处理；"
        "网格仅用于可视化。\n\n"
        "选定目标版建议图注：选定目标的三维定位与遗漏示例。每行使用相同真实标注目标，并为各方法展示与其匹配的代表预测"
        "（使用GT辅助匹配，三维AABB IoU>0.15）；未展示全部预测，包括部分误检。"
        "绿色、蓝色、橙色和金色分别表示GT、原生预测、PLR读出和CALR读出。场景网格仅用于可视化。\n\n"
        "本图来自20261009公开CLI协议实验保存的真实终态预测，最终路线为PLR-v1、CALR-v1及无visibility三因素历史最大HCSR。"
        "与独立主表运行的文件来源不同，不能把图中数量直接当作主表该场景的数量。"
        "HCSR保持原生几何不变，因此本图主要展示覆盖和定位；排序效果仍需分数/排名标注与严格定量对照。\n\n"
        "source_manifest.json记录输入文件、SHA256、相机、剖切、全部选择索引及坐标验证；"
        "*_measured_boxes.npz包含真实三维坐标和分数。没有移动、缩紧、补画预测框，也没有重新运行模型。\n",
        encoding="utf-8")
    if individual_exports:
        text = ["独立三维图索引\n", "所有PNG为3000×2400、400 DPI、透明背景，无标题和图例，便于论文排版。\n",
                "individual_selected：与清晰组合图相同的GT辅助选定目标，PNG/PDF/SVG。\n",
                "individual_top50：与Top-50组合图相同的分数读出，PNG。\n",
                "gt为真实标注；boxfusion为原版；boxfusion_ctrl为受控基线；recar3d为完整方法。\n",
                "每个场景的四张图共享网格、相机、坐标与裁剪；颜色和筛选说明见图注与使用说明.txt。\n\n"]
        for item in individual_exports:
            text.append(f"{item['scene']} | {item['method']} | {item['mode']} | "
                        f"选取{len(item['selected_indices'])}/{item['total_records']}条\n")
            text.extend(f"  {path}\n" for path in item["files"])
        (output / "独立图片索引.txt").write_text("".join(text), encoding="utf-8")
    with zipfile.ZipFile(output.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                archive.write(p, f"{output.name}/{p.relative_to(output)}")
    print(json.dumps({"output": str(output), "source": str(RUN), "scenes": [x["scene"] for x in specs],
                      "individual_panel_count": len(individual_exports)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
