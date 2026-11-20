#!/usr/bin/env python3
"""Add a measured CA-1M example to the ScanNet qualitative comparison.

Visualization only: reads existing world-coordinate meshes, GT and predictions;
does not run a detector or change predictions. Selection is explicitly GT-assisted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from plyfile import PlyData

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qualitative import optimize_qualitative_evidence as opt
from tools.qualitative import render_boxfusion_ctrl_recar_3d as original
from tools.qualitative import split_error_and_add_scene as split

OUT = ROOT / "figure_assets/qualitative_scannet_ca1m_20261102"
DATA = Path("/extra/ZhaoX/boxfusion_ca1m")
RUN = ROOT / "development/final_controls/experiments/ca1m_novis107_20261006"
SCENE = "42897692"
TARGET = 1
CONTEXT = [3, 4, 5, 7]


def make_camera(renderer, xyz, faces, azimuth, elevation=34., scale=1):
    width, height = 1500*scale, 1200*scale
    # World Z is already gravity aligned. Recentring is unnecessary and would
    # obscure provenance; use the original approximately 285-m height offset.
    visible = xyz[np.unique(faces)]
    look = (visible.min(0)+visible.max(0))/2
    az, el = np.deg2rad([azimuth, elevation])
    direction = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    position = look+16*direction
    principal = np.array([width/2, height/2], dtype=float)
    focal = 2000.*scale
    uv = renderer.project(visible, position, look, focal, principal)[:, :2]
    lower, upper = uv.min(0), uv.max(0)
    focal *= min(width*.9/(upper[0]-lower[0]), height*.88/(upper[1]-lower[1]))
    uv = renderer.project(visible, position, look, focal, principal)[:, :2]
    principal += np.array([width/2, height/2])-(uv.min(0)+uv.max(0))/2
    camera = {"position": position.tolist(), "look_at": look.tolist(),
              "focal_pixels": float(focal), "principal_point_pixels": principal.tolist(),
              "image_size": [width, height], "azimuth_degrees": azimuth,
              "elevation_degrees": elevation, "display_limits": [0, width, height, 0],
              "source": "Shared fixed Z-up perspective fitted to the actual mesh; no per-method change"}
    return camera, (position, look, focal, principal)


def load_scene(renderer, *, azimuth=-58., scale=2):
    folder = DATA/SCENE
    gt_path = folder/"after_filter_boxes.npy"
    gt = np.load(gt_path).astype(float)
    pred_sources = {
        "BoxFusion": ROOT/"results/full"/f"{SCENE}_boxes.pkl",
        "BoxFusion-Ctrl": RUN/"factorial_no_visibility/base"/f"{SCENE}_boxes.pkl",
        "ReCaR-3D": RUN/"factorial_no_visibility/p_a_m2"/f"{SCENE}_boxes.pkl",
    }
    predictions = {name: original.read_predictions(path, np.eye(4)) for name, path in pred_sources.items()}
    native = predictions["BoxFusion-Ctrl"]
    full = predictions["ReCaR-3D"]
    native_n = len(native["raw"])
    assert np.array_equal(native["raw"], full["raw"][:native_n])
    # Identify branch membership using the exported component arrays, not score.
    plr = original.read_predictions(RUN/"components/births_plr_v1"/f"{SCENE}_boxes.pkl", np.eye(4))
    calr = original.read_predictions(RUN/"components/births_calr_v1"/f"{SCENE}_boxes.pkl", np.eye(4))
    assert np.array_equal(full["raw"][native_n:native_n+len(plr["raw"])], plr["raw"])
    assert np.array_equal(full["raw"][native_n+len(plr["raw"]):], calr["raw"])
    for name, pred in predictions.items():
        pred["branches"] = (["native"]*native_n+["plr"]*len(plr["raw"])+["calr"]*len(calr["raw"])) if name == "ReCaR-3D" else ["native"]*len(pred["boxes"])
        pred["selected"] = original.selected_matches(pred["boxes"], gt, CONTEXT)
    matrices = {name: original.iou_matrix(pred["raw"], gt) for name, pred in predictions.items()}
    best = native_n+int(np.argmax(matrices["ReCaR-3D"][native_n:, TARGET]))
    assert full["branches"][best] == "calr"
    assert matrices["ReCaR-3D"][best, TARGET] >= .5
    assert max(matrices["BoxFusion"][:, TARGET].max(), matrices["BoxFusion-Ctrl"][:, TARGET].max()) <= .15
    full["selected"] = native["selected"].copy()+[best]
    recovery = {"gt_index": TARGET, "prediction_index": best, "branch": "VAR",
                "recovered_iou": float(matrices["ReCaR-3D"][best, TARGET]),
                "original_best_iou": float(matrices["BoxFusion"][:, TARGET].max()),
                "native_best_iou": float(matrices["BoxFusion-Ctrl"][:, TARGET].max())}

    mesh_path = folder/"mesh.ply"
    mesh = PlyData.read(mesh_path)
    v = mesh["vertex"]
    xyz = np.column_stack([v[k] for k in ("x", "y", "z")]).astype(float)
    colors = np.column_stack([v[k] for k in ("red", "green", "blue")]).astype(float)/255.
    faces = np.stack(mesh["face"].data["vertex_indices"]).astype(np.int32)
    assert np.isfinite(xyz).all() and faces.max() < len(xyz)
    # Do not cut away any furniture or inferred wall instances in CA-1M.
    # Use the full measured mesh and a common camera instead.
    camera, args = make_camera(renderer, xyz, faces, azimuth, scale=scale)
    bg = renderer.rasterize(renderer.project(xyz, *args), colors, faces, *camera["image_size"])
    # A shared viewport removes empty margins and tiny disconnected scan streaks
    # from the figure, without removing faces from the rendered mesh. Include
    # every displayed corner so the crop cannot hide a displayed detection.
    occupied = bg[:, :, 3] > 0
    xs = np.flatnonzero(occupied.sum(0) > camera["image_size"][1]*.02)
    ys = np.flatnonzero(occupied.sum(1) > camera["image_size"][0]*.015)
    shown = [gt[[TARGET]+CONTEXT].reshape(-1, 3)]
    shown.extend(pred["boxes"][pred["selected"]].reshape(-1, 3) for pred in predictions.values())
    projected_corners = renderer.project(np.concatenate(shown), *args)[:, :2]
    lower = np.minimum([xs[0], ys[0]], projected_corners.min(0))-55*scale
    upper = np.maximum([xs[-1], ys[-1]], projected_corners.max(0))+55*scale
    camera["display_limits"] = [float(lower[0]), float(upper[0]), float(upper[1]), float(lower[1])]
    camera["viewport_rule"] = "Common occupied-mesh viewport plus all displayed GT/prediction corners and 55-pixel base-scale margin; no mesh faces deleted"
    payload = {"spec": {"scene": SCENE, "dataset": "CA-1M", "gt_indices": [TARGET]+CONTEXT},
               "gt": gt, "predictions": predictions, "background": bg, "camera": camera,
               "projection_args": args, "recoveries": [recovery], "unresolved": [], "hcsr": None}
    audit = {"scene": SCENE, "dataset": "CA-1M", "gt_path": str(gt_path),
             "gt_sha256": opt.digest(gt_path), "gt_indices": [TARGET]+CONTEXT,
             "mesh": {"path": str(mesh_path), "sha256": opt.digest(mesh_path),
                      "vertices": len(xyz), "faces": len(faces), "removed_faces": 0,
                      "colors": "Unmodified measured per-vertex RGB"},
             "coordinate_transform": np.eye(4).tolist(),
             "coordinate_note": "GT, mesh and all prediction corners already share world coordinates and Z-up; no gravity transform applied a second time",
             "iou": "3D AABB IoU from world corner bounds, matching the CA-1M evaluator",
             "recoveries": [recovery], "camera": camera,
             "methods": {name: {"path": pred["path"], "sha256": pred["sha256"],
                                 "total": len(pred["boxes"]), "selected_indices": pred["selected"]}
                         for name, pred in predictions.items()},
             "baseline_provenance": "Existing BoxFusion results/full archive, the CA-1M output directory named in the repository README; not a fresh inference run or an independently reverified reproduction of the published AP",
             "final_configuration": "Existing CA-1M-107 no-visibility HCSR + current PLR + CALR-v1 factorial exports",
             "selection": "GT-assisted illustrative target and context; not random sampling or a complete prediction visualization",
             "native_geometry_identical_in_ctrl_and_full": True}
    return payload, audit


def previews(renderer):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, az in zip(axes.flat, [-140., -100., -58., -15., 40., 135.]):
        p, _ = load_scene(renderer, azimuth=az, scale=1)
        split.render_panel(ax, p, renderer, 3)
        opt.draw_box(ax, p["gt"][TARGET], renderer, p["projection_args"], opt.COLORS["gt"], 1., True)
        ax.set_title(f"CA-1M {SCENE} · azimuth {az:g}°", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT/"ca1m_view_candidates.png", dpi=140)
    plt.close(fig)


def render_comparison(payloads, renderer, stem):
    rows = len(payloads)
    fig, axes = plt.subplots(rows, 4, figsize=(opt.WIDTH, 2.18*rows+.4), squeeze=False)
    fig.subplots_adjust(left=.025, right=.99, bottom=.34/(2.18*rows+.4), top=1-.40/(2.18*rows+.4), wspace=.035, hspace=.20)
    split.column_titles(fig, axes[0], 1-.17/(2.18*rows+.4))
    for row, p in enumerate(payloads):
        item = p["recoveries"][0]
        for column, ax in enumerate(axes[row]):
            split.render_panel(ax, p, renderer, column)
            if column == 0:
                label, color, box = "Target", opt.COLORS["gt"], p["gt"][item["gt_index"]]
            elif column in (1, 2):
                label, color, box = "No match", opt.MUTED, p["gt"][item["gt_index"]]
            else:
                label = f"{item['branch']}\nIoU {item['recovered_iou']:.2f}"
                color = opt.COLORS["plr"] if item["branch"] == "PTR" else opt.COLORS["calr"]
                box = p["predictions"]["ReCaR-3D"]["boxes"][item["prediction_index"]]
            opt.pointer(ax, box, renderer, p["projection_args"], label, color, (.77, .88), 6.2)
        bounds = axes[row, 0].get_position()
        dataset = p["spec"].get("dataset", "ScanNet")
        letter = f"({chr(97+row)}) " if rows > 1 else ""
        fig.text(.025, bounds.y1+.011, f"{letter}{dataset} · {p['spec']['scene']} · {item['branch']} recovery",
                 fontsize=6.9, fontweight="bold", color=opt.INK)
    handles = [Line2D([], [], color=opt.COLORS[key], lw=1.6, ls="--" if key == "gt" else "-", label=label)
               for key, label in (("gt", "GT"), ("native", "Native box"), ("plr", "PTR recovery"), ("calr", "VAR recovery"))]
    height = 2.18*rows+.4
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .17/height), ncol=4,
               frameon=False, fontsize=6.5, columnspacing=1.7)
    fig.text(.5, .12/height, "GT-assisted selected targets; not all predictions. False positives and duplicates may be omitted.",
             ha="center", color=opt.MUTED, fontsize=5.9)
    fig.text(.5, .025/height, "Same camera per scene; native geometry unchanged. Scene mesh is used only for visualization.",
             ha="center", color=opt.MUTED, fontsize=5.9)
    return opt.export(fig, stem)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--azimuth", type=float, default=40.)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    opt.OUT, split.OUT = OUT, OUT
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "text.color": opt.INK})
    renderer = original.load_renderer()
    if args.preview:
        previews(renderer)
        return
    p, audit = load_scene(renderer, azimuth=args.azimuth)
    Image.fromarray(p["background"]).save(OUT/f"ca1m_{SCENE}_mesh.png")
    measured = {"gt_boxes": p["gt"]}
    for name, pred in p["predictions"].items():
        slug = original.METHOD_SLUGS[name]
        measured[slug+"_boxes"] = pred["raw"]
        measured[slug+"_scores"] = pred["scores"]
        measured[slug+"_selected_indices"] = np.array(pred["selected"], dtype=int)
    np.savez_compressed(OUT/f"ca1m_{SCENE}_measured_boxes.npz", **measured)
    input_manifest = opt.INPUT/"source_manifest.json"
    hashes = {str(input_manifest): opt.digest(input_manifest)}
    old_payloads, old_audits = opt.load_payloads(renderer, json.loads(input_manifest.read_text()), hashes)
    selected = old_payloads[:2]+[p]
    files = render_comparison(selected, renderer, "qualitative_3d_scannet_ca1m")
    files += render_comparison([p], renderer, f"ca1m_{SCENE}_comparison")
    individuals = []
    for payload in selected:
        individuals.extend(split.export_individuals(payload, renderer, OUT/"individuals"))
    for record in audit["methods"].values():
        hashes[record["path"]] = record["sha256"]
    hashes[audit["gt_path"]] = audit["gt_sha256"]
    hashes[audit["mesh"]["path"]] = audit["mesh"]["sha256"]
    manifest = {"sources_sha256": hashes, "scenes": old_audits[:2]+[audit],
                "individuals": individuals, "exports": files, "model_inference_performed": False,
                "GPU_used": False, "prediction_coordinates_modified": False,
                "PTR_VAR_code_names": {"PTR": "PLR", "VAR": "CALR"}}
    (OUT/"source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    caption = f"""图X｜ScanNet与CA-1M中选定目标的三维定性比较。
每行从左至右为选定GT、BoxFusion、BoxFusion-Ctrl和ReCaR-3D，采用相同的相机、网格及坐标系。绿色虚线表示GT，蓝色表示原生框，橙色和金色分别表示PTR和VAR恢复框。(a) ScanNet scene0426_03，PTR恢复目标的3D AABB IoU为0.58；(b) ScanNet scene0606_01，VAR恢复目标的IoU为0.86；(c) CA-1M {SCENE}，VAR恢复目标的IoU为{audit['recoveries'][0]['recovered_iou']:.2f}。该CA-1M目标与原版及受控基线全部输出的最大IoU均为0。
上述例子经过GT辅助选择，仅展示选定目标和部分上下文框，不代表完整预测或统一Top-K读出，误检及重复输出可能未展示，不能据此估计AP或误检率。ReCaR-3D保留Ctrl已展示的相同原生几何，再添加所选恢复框。场景网格仅用于可视化，不作为模型输入。

来源说明：CA-1M的BoxFusion列取自仓库既有results/full/{SCENE}_boxes.pkl归档；这是仓库README指定的CA-1M输出目录，并非本次重跑，也不声称本次重新复现了论文AP。Ctrl与ReCaR-3D来自ca1m_novis107_20261006/factorial_no_visibility的base和p_a_m2，最终配置为无visibility的HCSR、PLR及CALR-v1。分支来源由独立components文件逐框核对。CA-1M中GT、mesh.ply及预测已处于相同Z-up世界系，未重复应用重力矩阵，未裁切网格面片或更改预测。

文件：qualitative_3d_scannet_ca1m为2个ScanNet场景+1个CA-1M场景主图；ca1m_{SCENE}_comparison为CA-1M独立对照；individuals包含12张3000×2400、400 DPI透明背景PNG；measured_boxes和source_manifest保存原始坐标、所选索引及SHA256审计。
原独立错误分析图保持不变：../qualitative_success_error_split_20261102/error_analysis_persistent_miss.png。
"""
    (OUT/"图注与来源说明.txt").write_text(caption)
    archive = OUT.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                bundle.write(path, str(Path(OUT.name)/path.relative_to(OUT)))
    with zipfile.ZipFile(OUT/f"ca1m_{SCENE}_four_individuals.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted((OUT/"individuals").glob(f"{SCENE}_*.png")):
            bundle.write(path, path.name)
    print(json.dumps({"output": str(OUT), "archive": str(archive), "ca1m": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
