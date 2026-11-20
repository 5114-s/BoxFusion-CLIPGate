#!/usr/bin/env python3
"""Separate the persistent-miss example and add a real recovery scene.

Uses archived predictions and measured mesh geometry only; no model inference.
Selected-target examples are explicitly GT-assisted, not complete outputs.
"""
from __future__ import annotations

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
from tools.qualitative import optimize_qualitative_evidence as opt
from tools.qualitative import render_boxfusion_ctrl_recar_3d as original

OUT = ROOT/"figure_assets/qualitative_success_error_split_20261102"
NEW_SCENE = {"scene": "scene0300_00", "gt_indices": [2, 3, 5, 8], "walls": [14, 15]}


def add_scene(renderer, hashes):
    p = original.scene_payload(NEW_SCENE, renderer, OUT, raster_scale=2)
    pred, gt = p["predictions"], p["gt"]
    native = pred["BoxFusion-Ctrl"]
    n = len(native["boxes"])
    full = pred["ReCaR-3D"]
    matrices = {name: original.iou_matrix(value["boxes"], gt) for name, value in pred.items()}
    assert np.array_equal(native["raw"], full["raw"][:n])
    selected = native["selected"].copy()
    cases = []
    # Two DIFFERENT GT targets, one illustrative readout per candidate level.
    # Branch-specific selection is disclosed; it does not change any predictions.
    for target, branch in ((8, "PTR"), (2, "VAR")):
        allowed = np.array([b == ("plr" if branch == "PTR" else "calr") for b in full["branches"]])
        indices = np.flatnonzero(allowed)
        index = int(indices[np.argmax(matrices["ReCaR-3D"][indices, target])])
        native_overlap = float(matrices["BoxFusion-Ctrl"][:, target].max())
        original_overlap = float(matrices["BoxFusion"][:, target].max())
        overlap = float(matrices["ReCaR-3D"][index, target])
        assert max(native_overlap, original_overlap) <= .15 and overlap >= .5
        assert index >= n and index not in selected
        selected.append(index)
        cases.append({"gt_index": target, "prediction_index": index, "branch": branch,
            "native_best_iou": native_overlap, "original_best_iou": original_overlap,
            "recovered_iou": overlap})
    full["selected"] = selected
    p["recoveries"], p["unresolved"], p["hcsr"] = cases, [], None
    p["audit"]["methods"]["ReCaR-3D"]["selected_indices"] = selected
    # Keep the saved index metadata consistent with the native-preserving view.
    measured_path = OUT/f"{NEW_SCENE['scene']}_measured_boxes.npz"
    with np.load(measured_path) as z:
        measured = {key: z[key] for key in z.files}
    measured["recar_selected_indices"] = np.asarray(selected, dtype=int)
    assert np.array_equal(measured["recar_boxes"], full["boxes"])
    np.savez_compressed(measured_path, **measured)
    for data in p["audit"]["methods"].values():
        hashes[data["path"]] = data["sha256"]
    hashes[p["audit"]["gt_path"]] = p["audit"]["gt_sha256"]
    hashes[p["audit"]["mesh"]["path"]] = p["audit"]["mesh"]["sha256"]
    audit = {"scene": NEW_SCENE["scene"], "selected_gt_indices": NEW_SCENE["gt_indices"],
        "method_selected_indices": {name: value["selected"] for name, value in pred.items()},
        "same_selected_native_boxes_in_ctrl_and_full": True, "recoveries": cases,
        "camera": p["camera"], "mesh": p["audit"]["mesh"],
        "GT_assisted_branch_specific_selection": True,
        "selection_note": "Archived scenes inspected for illustrative native misses with recovery IoU>=0.50; not random scene sampling or a new test benchmark"}
    return p, audit


def render_panel(ax, p, renderer, column, mode="selected"):
    opt.base_axis(ax, p)
    method = original.METHODS[column]
    if column == 0:
        indices = p["spec"]["gt_indices"] if mode == "selected" else range(len(p["gt"]))
        for index in indices:
            opt.draw_box(ax, p["gt"][index], renderer, p["projection_args"], opt.COLORS["gt"], .9, True)
        count = f"{len(indices)} selected GT" if mode == "selected" else f"{len(indices)} GT"
    else:
        data = p["predictions"][method]
        indices = data["selected"] if mode == "selected" else np.argsort(-data["scores"], kind="stable")[:50]
        for index in indices:
            branch = data["branches"][index]
            opt.draw_box(ax, data["boxes"][index], renderer, p["projection_args"], opt.COLORS[branch],
                         .8 if branch == "native" else 1.3)
        count = f"{len(indices)} shown / {len(data['boxes'])} total"
    ax.text(.5, -.035, count, transform=ax.transAxes, ha="center", va="top",
            color=opt.MUTED, fontsize=5.9)


def column_titles(fig, axes, y, *, all_gt=False):
    for column, name in enumerate(("GT (all)" if all_gt else "GT (selected)", *original.METHODS[1:])):
        bounds = axes[column].get_position()
        fig.text(bounds.x0+bounds.width/2, y, name, ha="center", fontsize=8.1,
                 color=opt.INK, fontweight="bold")


def render_success(payloads, renderer):
    fig, axes = plt.subplots(3, 4, figsize=(opt.WIDTH, 6.85))
    fig.subplots_adjust(left=.025, right=.99, bottom=.12, top=.925, wspace=.035, hspace=.25)
    column_titles(fig, axes[0], .970)
    titles = ("PTR recovery", "VAR recovery", "PTR and VAR recover different targets")
    for row, p in enumerate(payloads):
        for column, method in enumerate(original.METHODS):
            ax = axes[row, column]
            render_panel(ax, p, renderer, column)
            for position, item in enumerate(p["recoveries"]):
                location = (.77, .86) if row < 2 else ((.75, .9) if position == 0 else (.72, .20))
                target = p["gt"][item["gt_index"]]
                if column == 0:
                    label = "Target" if row < 2 else "Target A" if position == 0 else "Target B"
                    color, box = opt.COLORS["gt"], target
                elif column < 3:
                    label, color, box = "No match", opt.MUTED, target
                    if row == 2:
                        matrix = original.iou_matrix(p["predictions"][method]["boxes"], p["gt"])
                        assert matrix[:, item["gt_index"]].max() <= .15
                else:
                    label = f"{item['branch']}\nIoU {item['recovered_iou']:.2f}"
                    color = opt.COLORS["plr"] if item["branch"] == "PTR" else opt.COLORS["calr"]
                    box = p["predictions"][method]["boxes"][item["prediction_index"]]
                opt.pointer(ax, box, renderer, p["projection_args"], label, color, location, 6.2)
        bounds = axes[row, 0].get_position()
        fig.text(.025, bounds.y1+.012, f"({chr(97+row)}) {p['spec']['scene']} · {titles[row]}",
                 fontsize=6.9, fontweight="bold", color=opt.INK)
    handles = [Line2D([], [], color=opt.COLORS[key], lw=1.6, ls="--" if key == "gt" else "-", label=label)
        for key, label in (("gt", "GT"), ("native", "Native box"), ("plr", "PTR recovery"), ("calr", "VAR recovery"))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .057), ncol=4,
               frameon=False, fontsize=6.7, columnspacing=1.8)
    fig.text(.5, .037, "GT-assisted selected targets; not all predictions. Omitted outputs include false positives and duplicates.",
             ha="center", color=opt.MUTED, fontsize=6.1)
    fig.text(.5, .019, "Identical perspective per scene. Native geometry is unchanged; mesh is used only for visualization.",
             ha="center", color=opt.MUTED, fontsize=6.1)
    return opt.export(fig, "qualitative_3d_success_three_scenes")


def render_error(p, renderer):
    fig, axes = plt.subplots(1, 4, figsize=(opt.WIDTH, 2.78))
    fig.subplots_adjust(left=.025, right=.99, bottom=.30, top=.84, wspace=.035)
    column_titles(fig, axes, .955)
    fig.text(.025, .865, f"{p['spec']['scene']} · Persistent missed target", fontsize=7.6,
             fontweight="bold", color=opt.INK)
    missed = p["unresolved"][0]
    for column, ax in enumerate(axes):
        render_panel(ax, p, renderer, column)
        if column in (0, 3):
            opt.pointer(ax, p["gt"][missed["gt_index"]], renderer, p["projection_args"],
                        "Missed target" if column == 0 else "Unrecovered GT", opt.COLORS["failure"], (.75, .9), 6.4)
        if column == 3:
            item = p["hcsr"]
            opt.pointer(ax, p["predictions"]["ReCaR-3D"]["boxes"][item["native_index"]],
                renderer, p["projection_args"], "HCSR\nrank 4 → 3", opt.COLORS["hcsr"], (.23, .18), 6.3)
    fig.text(.5, .19, f"Unrecovered target: maximum IoU over all ReCaR-3D predictions = {missed['full_best_iou']:.3f} < 0.15.",
             ha="center", color=opt.COLORS["failure"], fontsize=7.0)
    fig.text(.5, .12, "HCSR improves another native box's rank; it does not create a detection for the missing target.",
             ha="center", color=opt.MUTED, fontsize=6.5)
    fig.text(.5, .05, "Green dashed: GT · Blue: native predictions · Purple: HCSR callout. GT-assisted selected-target view.",
             ha="center", color=opt.MUTED, fontsize=6.3)
    return opt.export(fig, "error_analysis_persistent_miss")


def export_individuals(p, renderer, folder):
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for column, name in enumerate(original.METHODS):
        fig = plt.figure(figsize=(7.5, 6), dpi=400)
        ax = fig.add_axes([0, 0, 1, 1]); opt.base_axis(ax, p)
        if column == 0:
            indices = p["spec"]["gt_indices"]
            for index in indices:
                opt.draw_box(ax, p["gt"][index], renderer, p["projection_args"], opt.COLORS["gt"], 1.2, True)
        else:
            data = p["predictions"][name]; indices = data["selected"]
            for index in indices:
                branch = data["branches"][index]
                opt.draw_box(ax, data["boxes"][index], renderer, p["projection_args"], opt.COLORS[branch],
                             1.1 if branch == "native" else 1.65)
        path = folder/f"{p['spec']['scene']}_{original.METHOD_SLUGS[name]}.png"
        fig.savefig(path, dpi=400, transparent=True); plt.close(fig)
        records.append({"scene": p["spec"]["scene"], "method": name,
                        "indices": [int(i) for i in indices], "file": str(path.relative_to(OUT)),
                        "sha256": opt.digest(path), "size": [3000, 2400]})
    return records


def new_scene_audit(p, renderer):
    fig, axes = plt.subplots(1, 4, figsize=(opt.WIDTH, 2.5))
    fig.subplots_adjust(left=.025, right=.99, bottom=.24, top=.85, wspace=.035)
    column_titles(fig, axes, .95, all_gt=True)
    for column, ax in enumerate(axes):
        render_panel(ax, p, renderer, column, mode="top50")
    fig.text(.5, .115, f"{p['spec']['scene']} · At most 50 predictions per method, by output score; no extra NMS.",
             ha="center", color=opt.MUTED, fontsize=6.4)
    fig.text(.5, .05, "BoxFusion archive scores are all tied; archived row order is used. This audit is not GT-selected.",
             ha="center", color=opt.MUTED, fontsize=6.2)
    # The GT header in this audit shows all GT, not the selected subset.
    axes[0].set_title("", pad=0)
    return opt.export(fig, "new_scene_top50_audit")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    opt.OUT = OUT
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                        "svg.fonttype": "none", "text.color": opt.INK})
    renderer = original.load_renderer()
    manifest_path = opt.INPUT/"source_manifest.json"
    hashes = {str(manifest_path): opt.digest(manifest_path)}
    old_payloads, old_audits = opt.load_payloads(renderer, json.loads(manifest_path.read_text()), hashes)
    fresh, new_audit = add_scene(renderer, hashes)
    success = old_payloads[:2]+[fresh]
    error = old_payloads[2]
    files = render_success(success, renderer)+render_error(error, renderer)+new_scene_audit(fresh, renderer)
    individuals = []
    for payload in success:
        individuals.extend(export_individuals(payload, renderer, OUT/"main_individuals"))
        print(f"success scene exported: {payload['spec']['scene']}", flush=True)
    individuals.extend(export_individuals(error, renderer, OUT/"error_individuals"))
    caption = """主图：三场景恢复结果；错误分析独立成图（2026-11-02）

定性主图：qualitative_3d_success_three_scenes.png/.pdf/.svg
图X｜三个场景中选定目标的三维定性比较。每行从左至右为选定GT、BoxFusion、BoxFusion-Ctrl和ReCaR-3D，使用相同相机、网格剖切与坐标系。绿色虚线为GT，蓝色为原生预测，橙色和金色分别为PTR与VAR恢复框。(a) scene0426_03中PTR恢复目标的3D AABB IoU为0.58；(b) scene0606_01中VAR恢复目标的IoU为0.86；(c) 新增scene0300_00，PTR与VAR分别覆盖两个不同的遗漏目标，IoU分别为0.69和0.58。新增两目标与原版/受控基线所有框的最大IoU均不超过0.15。主图是GT辅助选定目标和分支代表预测的示例，ReCaR-3D优先保留与Ctrl相同的已展示原生几何，再添加指定恢复框，不表示完整输出或统一Top-K结果，部分误检及重复框未展示。不得凭此推断误检率或整体AP。场景网格仅用于可视化。

错误分析图：error_analysis_persistent_miss.png/.pdf/.svg
图Y｜未被恢复的目标及HCSR作用边界。scene0432_01中，红色箭头指向的GT与完整ReCaR-3D全部预测的最大3D AABB IoU约为0.054，低于0.15评测阈值，因此该目标仍未被匹配。紫色标注指向另一个原生框，其分数由0.825升至0.909，场景内原生排名由4升至3；HCSR不改变该框几何，也不会自动为红色箭头处创建新检测。每列只显示GT辅助选定的代表框，不是全部输出。该图说明当前候选历史仍不能保证完整目标覆盖，而不是HCSR几何校正失败。

正文可用表述：
图X展示了PTR和VAR在三个场景中补充原生输出目标覆盖的实例。新增场景中，两层恢复分别覆盖不同的遗漏目标。图Y单独展示持续遗漏：即使HCSR提升了另一原生框的排名，仍有GT未被完整系统覆盖。可视化采用GT辅助选择，整体准确率、排序和误检表现以定量实验为依据。

素材：main_individuals内为主图12张独立三维PNG；error_individuals内为错误图4张独立PNG。全部为3000×2400、400 DPI、透明背景。new_scene_top50_audit是新增场景按各方法原有分数逐场景Top-50的审计图；原版BoxFusion分数全为1，稳定使用存档行序。主图与Top-50审计的选择规则不同，不混用。

命名与来源：PTR/VAR分别对应既有实现中的PLR/CALR。所有预测均来自20261009公开CLI复现存档，并不重新生成模型结果。新增场景通过查看已存预测与GT覆盖后选择，属于定性成功示例筛选，不是随机场景或新的盲测。新增场景移除wall实例14、15以及既有统一规则确定的顶面，仅用于可视化；没有删除家具或修改预测框。每个场景的四个方法始终共享相同剖切和视角。原有优化图、时序图及HCSR成功/失败图均未覆盖或删除。
"""
    (OUT/"图注与使用说明.txt").write_text(caption, encoding="utf-8")
    manifest = {"schema": "recar3d.qualitative.success_error_split.v1",
        "success_scenes": [p["spec"]["scene"] for p in success],
        "error_scene": error["spec"]["scene"], "new_scene_audit": new_audit,
        "success_audits": old_audits[:2]+[new_audit], "error_audit": old_audits[2],
        "individuals": individuals, "input_sha256": hashes,
        "script_sha256": opt.digest(Path(__file__)),
        "artifacts_sha256": {name: opt.digest(OUT/name) for name in files},
        "GT_assisted_selection": True, "prediction_coordinates_edited": False,
        "new_model_inference": False, "GPU_used": False,
        "preserved_original_assets": True}
    (OUT/"source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    with zipfile.ZipFile(OUT.with_suffix(".zip"), "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file(): archive.write(path, path.relative_to(OUT))
    print(json.dumps({"output": str(OUT), "main_scenes": manifest["success_scenes"],
                      "error_scene": manifest["error_scene"], "individual_images": len(individuals)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
