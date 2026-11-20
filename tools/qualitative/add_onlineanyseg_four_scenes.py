#!/usr/bin/env python3
"""Add scene0300_00 and compact the measured five-method 3D comparison.

The script does not run a detector and does not edit any prediction geometry.
It loads the already-exported OnlineAnySeg final instance masks converted to
world-coordinate AABBs, then uses the same explicitly GT-assisted illustrative
selection protocol, scene mesh, camera and viewport as the four-column figure.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qualitative import add_ca1m_scene as ca1m
from tools.qualitative import optimize_qualitative_evidence as opt
from tools.qualitative import render_boxfusion_ctrl_recar_3d as original
from tools.qualitative import split_error_and_add_scene as split

INPUT_FIGURE = ROOT / "figure_assets/qualitative_scannet_ca1m_20261102"
OAS_ROOT = ROOT / "external/OnlineAnySeg_runs_20261102/aabbs_o3d019"
OUT = ROOT / "figure_assets/qualitative_scannet_ca1m_onlineanyseg_four_scenes_20261105"
SCENES = ("scene0426_03", "scene0606_01", "42897692", "scene0300_00")
OAS_COLOR = "#008F9C"
METHODS = ("Ground truth", "OnlineAnySeg", "BoxFusion", "BoxFusion-Ctrl", "ReCaR-3D")
SLUGS = {"Ground truth": "gt", "BoxFusion": "boxfusion",
         "BoxFusion-Ctrl": "boxfusion_ctrl", "OnlineAnySeg": "onlineanyseg",
         "ReCaR-3D": "recar3d"}
DISPLAY_NAMES = {"BoxFusion-Ctrl": "BoxFusion*", "ReCaR-3D": "CaRR-3D"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_payloads(renderer):
    # The CA-1M four-column manifest intentionally stores a compact audit of
    # its two reused ScanNet rows.  Reload those rows from their original,
    # fully traceable source manifest, as add_ca1m_scene.py does.
    source_manifest = opt.INPUT / "source_manifest.json"
    manifest = json.loads(source_manifest.read_text())
    hashes = {str(source_manifest): digest(source_manifest)}
    scan_payloads, scan_audits = opt.load_payloads(renderer, manifest, hashes)
    ca_payload, ca_audit = ca1m.load_scene(renderer, azimuth=40., scale=2)
    # scene0300_00 is the complementary case in which PTR and VAR recover
    # different native misses.  Rebuild it from the same archived predictions
    # and fixed-view renderer used by the audited four-column figure.
    split.OUT = OUT
    scene0300, scene0300_audit = split.add_scene(renderer, hashes)
    scene0300["spec"]["dataset"] = "ScanNet"
    payloads = scan_payloads[:2] + [ca_payload, scene0300]
    audits = scan_audits[:2] + [ca_audit, scene0300_audit]

    for payload, audit in zip(payloads, audits):
        scene = payload["spec"]["scene"]
        path = OAS_ROOT / f"{scene}.npz"
        json_path = OAS_ROOT / f"{scene}.json"
        with np.load(path) as data:
            boxes = np.asarray(data["world_boxes"], dtype=float)
            scores = np.asarray(data["scores"], dtype=float)
            point_counts = np.asarray(data["point_counts"], dtype=int)
        selected = original.selected_matches(boxes, payload["gt"], payload["spec"]["gt_indices"])
        payload["predictions"]["OnlineAnySeg"] = {
            "boxes": boxes,
            "scores": scores,
            "point_counts": point_counts,
            "selected": selected,
            "branches": ["onlineanyseg"] * len(boxes),
            "path": str(path),
            "sha256": digest(path),
        }
        overlaps = original.iou_matrix(boxes, payload["gt"])
        payload["onlineanyseg_targets"] = []
        for recovery in payload["recoveries"]:
            target = recovery["gt_index"]
            best_index = int(np.argmax(overlaps[:, target]))
            payload["onlineanyseg_targets"].append({
                "gt_index": int(target),
                "prediction_index": best_index,
                "best_iou": float(overlaps[best_index, target]),
                "matched_at_0.15": bool(overlaps[best_index, target] > .15),
            })
        hashes[str(path)] = digest(path)
        hashes[str(json_path)] = digest(json_path)
        audit["OnlineAnySeg"] = {
            "aabb_path": str(path),
            "aabb_sha256": hashes[str(path)],
            "conversion_audit_path": str(json_path),
            "conversion_audit_sha256": hashes[str(json_path)],
            "total_instances": int(len(boxes)),
            "selected_indices": [int(i) for i in selected],
            "targets": payload["onlineanyseg_targets"],
            "selection": "One-to-one maximum-IoU matches to the same GT-assisted selected targets; threshold 0.15",
        }
    return payloads, audits, hashes


def draw_method(ax, payload, renderer, method, *, show_count=True):
    opt.base_axis(ax, payload)
    if method == "Ground truth":
        indices = payload["spec"]["gt_indices"]
        for index in indices:
            opt.draw_box(ax, payload["gt"][index], renderer, payload["projection_args"],
                         opt.COLORS["gt"], .9, True)
        count = f"{len(indices)} selected GT"
    else:
        data = payload["predictions"][method]
        indices = data["selected"]
        for index in indices:
            if method == "OnlineAnySeg":
                color, width = OAS_COLOR, 1.05
            else:
                branch = data["branches"][index]
                color = opt.COLORS[branch]
                width = .8 if branch == "native" else 1.3
            opt.draw_box(ax, data["boxes"][index], renderer, payload["projection_args"],
                         color, width)
        count = f"{len(indices)} shown / {len(data['boxes'])} total"
    if show_count:
        ax.text(.5, -.035, count, transform=ax.transAxes, ha="center", va="top",
                color=opt.MUTED, fontsize=5.5)


def target_callout(ax, payload, renderer, method):
    """Annotate every complementary recovery target in a scene."""
    recoveries = payload["recoveries"]
    locations = ((.77, .88),) if len(recoveries) == 1 else ((.76, .88), (.72, .20))
    for position, (recovery, location) in enumerate(zip(recoveries, locations)):
        target = payload["gt"][recovery["gt_index"]]
        suffix = "" if len(recoveries) == 1 else f" {chr(65 + position)}"
        if method == "Ground truth":
            label, color, box = f"Target{suffix}", opt.COLORS["gt"], target
        elif method in ("BoxFusion", "BoxFusion-Ctrl"):
            label, color, box = "No match", opt.MUTED, target
        elif method == "OnlineAnySeg":
            item = payload["onlineanyseg_targets"][position]
            if item["matched_at_0.15"]:
                label = f"IoU {item['best_iou']:.2f}"
                color = OAS_COLOR
                box = payload["predictions"][method]["boxes"][item["prediction_index"]]
            else:
                label, color, box = "No match", opt.MUTED, target
        else:
            label = f"{recovery['branch']}\nIoU {recovery['recovered_iou']:.2f}"
            color = opt.COLORS["plr"] if recovery["branch"] == "PTR" else opt.COLORS["calr"]
            box = payload["predictions"][method]["boxes"][recovery["prediction_index"]]
        opt.pointer(ax, box, renderer, payload["projection_args"], label, color, location, 5.6)


def render_main(payloads, renderer):
    rows = len(payloads)
    height = 1.34 * rows + .46
    fig, axes = plt.subplots(rows, 5, figsize=(opt.WIDTH, height), squeeze=False)
    fig.subplots_adjust(left=.018, right=.995, bottom=.38/height, top=1-.42/height,
                        wspace=.018, hspace=.045)
    for column, method in enumerate(METHODS):
        bounds = axes[0, column].get_position()
        title = "GT (selected)" if method == "Ground truth" else DISPLAY_NAMES.get(method, method)
        fig.text(bounds.x0+bounds.width/2, 1-.16/height, title, ha="center",
                 fontsize=7.1, color=opt.INK, fontweight="bold")
    for row, payload in enumerate(payloads):
        for column, method in enumerate(METHODS):
            ax = axes[row, column]
            draw_method(ax, payload, renderer, method)
            target_callout(ax, payload, renderer, method)
        bounds = axes[row, 0].get_position()
        recovery = payload["recoveries"][0]
        recovery_label = "PTR + VAR" if len(payload["recoveries"]) > 1 else recovery["branch"]
        dataset = payload["spec"].get("dataset", "ScanNet")
        fig.text(.018, bounds.y1+.010,
                 f"({chr(97+row)}) {dataset} · {payload['spec']['scene']} · {recovery_label} recovery",
                 fontsize=6.3, fontweight="bold", color=opt.INK)
    handles = [
        Line2D([], [], color=opt.COLORS["gt"], lw=1.5, ls="--", label="GT"),
        Line2D([], [], color=opt.COLORS["native"], lw=1.5, label="Native box"),
        Line2D([], [], color=OAS_COLOR, lw=1.5, label="OnlineAnySeg"),
        Line2D([], [], color=opt.COLORS["plr"], lw=1.5, label="PTR recovery"),
        Line2D([], [], color=opt.COLORS["calr"], lw=1.5, label="VAR recovery"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .17/height),
               ncol=5, frameon=False, fontsize=5.9, columnspacing=1.2, handlelength=1.6)
    fig.text(.5, .105/height,
             "GT-assisted selected targets; not all predictions. OnlineAnySeg boxes are measured final-mask AABBs.",
             ha="center", color=opt.MUTED, fontsize=5.4)
    fig.text(.5, .025/height,
             "Same camera and world coordinates per scene; scene mesh is used only for visualization.",
             ha="center", color=opt.MUTED, fontsize=5.4)
    return opt.export(fig, "qualitative_3d_scannet_ca1m_with_onlineanyseg")


def render_four_column_layout(payloads, renderer, *, annotated):
    """Paper-style rows with the original and controlled BoxFusion baselines."""
    methods = ("Ground truth", "OnlineAnySeg", "BoxFusion", "BoxFusion-Ctrl", "ReCaR-3D")
    rows = len(payloads)
    # Axes are sized near the actual projected-scene aspect ratio.  This avoids
    # the large vertical blanks produced when equal-aspect panels sit inside
    # overly tall subplot rows.
    height = 1.34 * rows + .46
    fig, axes = plt.subplots(rows, len(methods), figsize=(opt.WIDTH, height), squeeze=False)
    fig.subplots_adjust(left=.055, right=.995, bottom=.31/height, top=1-.34/height,
                        wspace=.012, hspace=.045)

    for column, title in enumerate(("GT", "OnlineAnySeg", "BoxFusion", "BoxFusion*", "CaRR-3D (Ours)")):
        bounds = axes[0, column].get_position()
        fig.text(bounds.x0+bounds.width/2, 1-.12/height, title, ha="center",
                 fontsize=8.0, fontweight="bold", color=opt.INK)

    for row, payload in enumerate(payloads):
        for column, method in enumerate(methods):
            ax = axes[row, column]
            draw_method(ax, payload, renderer, method, show_count=False)
            if annotated:
                target_callout(ax, payload, renderer, method)
        bounds = axes[row, 0].get_position()
        dataset = payload["spec"].get("dataset", "ScanNet")
        fig.text(.018, bounds.y0+bounds.height/2,
                 f"({chr(97+row)})  {dataset}\n{payload['spec']['scene']}",
                 rotation=90, ha="center", va="center", fontsize=6.5,
                 fontweight="bold", color=opt.INK, linespacing=1.18)

    handles = [
        Line2D([], [], color=opt.COLORS["gt"], lw=1.5, ls="--", label="GT"),
        Line2D([], [], color=OAS_COLOR, lw=1.5, label="OnlineAnySeg"),
        Line2D([], [], color=opt.COLORS["native"], lw=1.5, label="Native box"),
        Line2D([], [], color=opt.COLORS["plr"], lw=1.5, label="PTR recovery"),
        Line2D([], [], color=opt.COLORS["calr"], lw=1.5, label="VAR recovery"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .105/height),
               ncol=5, frameon=False, fontsize=5.9, columnspacing=1.15,
               handlelength=1.55)
    fig.text(.52, .025/height,
             "GT-assisted selected targets; identical camera and world coordinates per row. Scene mesh is used only for visualization.",
             ha="center", color=opt.MUTED, fontsize=5.25)
    suffix = "annotated" if annotated else "clean"
    return opt.export(fig, f"qualitative_3d_layout_gt_onlineanyseg_boxfusion_boxfusionstar_carr_{suffix}")


def export_individuals(payloads, renderer):
    folder = OUT / "individuals"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for payload in payloads:
        for method in METHODS:
            fig = plt.figure(figsize=(7.5, 6), dpi=400)
            ax = fig.add_axes([0, 0, 1, 1])
            draw_method(ax, payload, renderer, method)
            target_callout(ax, payload, renderer, method)
            path = folder / f"{payload['spec']['scene']}_{SLUGS[method]}.png"
            fig.savefig(path, dpi=400, transparent=True)
            plt.close(fig)
            records.append({"scene": payload["spec"]["scene"], "method": method,
                            "path": str(path.relative_to(OUT)), "sha256": digest(path),
                            "size": [3000, 2400]})
    return records


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    opt.OUT = OUT
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "text.color": opt.INK})
    renderer = original.load_renderer()
    payloads, audits, hashes = load_payloads(renderer)
    exports = render_main(payloads, renderer)
    exports += render_four_column_layout(payloads, renderer, annotated=False)
    exports += render_four_column_layout(payloads, renderer, annotated=True)
    individuals = export_individuals(payloads, renderer)
    protocol = {
        "schema": "carr3d.qualitative.onlineanyseg.four_scenes.v1",
        "display_name": "CaRR-3D",
        "legacy_internal_method_key": "ReCaR-3D",
        "scenes": audits,
        "input_sha256": hashes,
        "exports": exports,
        "individuals": individuals,
        "official_onlineanyseg_inference_performed": True,
        "prediction_coordinates_modified": False,
        "box_conversion": "Coordinate-wise AABB of every official final instance mask; ScanNet axisAlignment applied once; CA-1M already world aligned",
        "selection": "GT-assisted illustrative one-to-one matching to the same selected targets; not complete output or Top-K",
        "onlineanyseg_scores": "Official final export supplies tied scores; no score or ranking was invented",
        "open3d_runtime": (
            "Existing three rows retain their audited 0.19.0 compatibility outputs; "
            "scene0300_00 was fused with upstream-requested Open3D 0.18.0 on CPU. "
            "Only device-string/checkpoint loading compatibility was changed."
        ),
    }
    (OUT / "source_manifest.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+"\n")
    caption = """图X｜ScanNet和CA-1M中选定目标的三维定性比较。
五列论文版从左至右为GT、OnlineAnySeg、BoxFusion、BoxFusion*和CaRR-3D。BoxFusion表示原始公开实现或结果归档；BoxFusion*表示与CaRR-3D共享原生候选生成和输出协议、但不启用候选历史模块的受控基线（内部实验键为BoxFusion-Ctrl）。前三行分别展示ScanNet上的PTR恢复、ScanNet上的VAR恢复和CA-1M上的VAR恢复；新增第四行scene0300_00同时展示PTR与VAR恢复两个不同的原生遗漏目标。所有列使用相同的世界坐标、场景网格、相机和视口。绿色虚线为GT，蓝色表示BoxFusion与BoxFusion*的原生框，青色为OnlineAnySeg最终实例掩码派生的三维AABB，橙色和金色分别为PTR与VAR恢复框。带标注版本中的数值为选定目标的实测3D AABB IoU；“No match”表示该方法全部输出在IoU阈值0.15下均未匹配该目标。

OnlineAnySeg采用官方CropFormer、CLIP、FCGF和三维融合流程；每个最终实例框由其导出的三维实例掩码全部点按坐标最小值/最大值生成。ScanNet框仅应用一次官方axisAlignment，CA-1M框保持原世界坐标。未人为赋予置信度或改变框几何。示例经过GT辅助选择，仅展示与同一组选定GT一对一匹配的代表框及部分上下文，不是完整预测、统一Top-K或定量准确率证明；整体性能应以AP实验为准。场景网格仅用于可视化。

文件说明：qualitative_3d_layout_gt_onlineanyseg_boxfusion_boxfusionstar_carr_clean为紧凑主图版；qualitative_3d_layout_gt_onlineanyseg_boxfusion_boxfusionstar_carr_annotated为带IoU和No match标注版；qualitative_3d_scannet_ca1m_with_onlineanyseg为同一五列比较的审计版。图中模型名称保持为GT、OnlineAnySeg、BoxFusion、BoxFusion*和CaRR-3D (Ours)。源代码中的ReCaR-3D与BoxFusion-Ctrl仅作为旧实验记录键保留。
"""
    (OUT / "图注与协议说明.txt").write_text(caption, encoding="utf-8")
    with zipfile.ZipFile(OUT.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUT))
    print(json.dumps({"output": str(OUT), "scenes": list(SCENES),
                      "individuals": len(individuals), "exports": exports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
