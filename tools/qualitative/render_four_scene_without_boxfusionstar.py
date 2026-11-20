#!/usr/bin/env python3
"""Render the four-scene qualitative figure without the BoxFusion* column.

This is a layout-only export.  It reuses the audited scene payloads, cameras,
selected boxes, and colors from add_onlineanyseg_four_scenes.py; no prediction
geometry or model output is changed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.qualitative import add_onlineanyseg_four_scenes as source


OUT = ROOT / "figure_assets/qualitative_scannet_ca1m_onlineanyseg_four_columns_20261107"
METHODS = ("Ground truth", "OnlineAnySeg", "BoxFusion", "ReCaR-3D")
TITLES = ("GT", "OnlineAnySeg", "BoxFusion", "CaRR-3D (Ours)")


def render(payloads, renderer, *, annotated: bool):
    rows = len(payloads)
    height = 1.34 * rows + .46
    fig, axes = plt.subplots(rows, len(METHODS),
                             figsize=(source.opt.WIDTH, height), squeeze=False)
    fig.subplots_adjust(left=.055, right=.995, bottom=.31/height,
                        top=1-.34/height, wspace=.014, hspace=.045)

    for column, title in enumerate(TITLES):
        bounds = axes[0, column].get_position()
        fig.text(bounds.x0 + bounds.width/2, 1-.12/height, title,
                 ha="center", fontsize=8.0, fontweight="bold",
                 color=source.opt.INK)

    for row, payload in enumerate(payloads):
        for column, method in enumerate(METHODS):
            ax = axes[row, column]
            source.draw_method(ax, payload, renderer, method, show_count=False)
            if annotated:
                source.target_callout(ax, payload, renderer, method)

        bounds = axes[row, 0].get_position()
        dataset = payload["spec"].get("dataset", "ScanNet")
        fig.text(.018, bounds.y0 + bounds.height/2,
                 f"({chr(97+row)})  {dataset}\n{payload['spec']['scene']}",
                 rotation=90, ha="center", va="center", fontsize=6.5,
                 fontweight="bold", color=source.opt.INK, linespacing=1.18)

    handles = [
        Line2D([], [], color=source.opt.COLORS["gt"], lw=1.5, ls="--", label="GT"),
        Line2D([], [], color=source.OAS_COLOR, lw=1.5, label="OnlineAnySeg"),
        Line2D([], [], color=source.opt.COLORS["native"], lw=1.5, label="Native box"),
        Line2D([], [], color=source.opt.COLORS["plr"], lw=1.5, label="PTR recovery"),
        Line2D([], [], color=source.opt.COLORS["calr"], lw=1.5, label="VAR recovery"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .105/height),
               ncol=5, frameon=False, fontsize=5.9, columnspacing=1.15,
               handlelength=1.55)
    fig.text(.52, .025/height,
             "GT-assisted selected targets; identical camera and world coordinates per row. "
             "Scene mesh is used only for visualization.",
             ha="center", color=source.opt.MUTED, fontsize=5.25)

    suffix = "annotated" if annotated else "clean"
    return source.opt.export(
        fig,
        f"qualitative_3d_gt_onlineanyseg_boxfusion_carr_four_scenes_{suffix}",
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source.OUT = OUT
    source.opt.OUT = OUT
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "text.color": source.opt.INK})

    renderer = source.original.load_renderer()
    payloads, audits, hashes = source.load_payloads(renderer)
    exports = []
    exports += render(payloads, renderer, annotated=False)
    exports += render(payloads, renderer, annotated=True)

    manifest = {
        "schema": "carr3d.qualitative.four_columns.no_boxfusionstar.v1",
        "layout_only_change": True,
        "methods": list(TITLES),
        "removed_column": "BoxFusion* (internal key: BoxFusion-Ctrl)",
        "scenes": audits,
        "input_sha256": hashes,
        "exports": exports,
        "selection": "Unchanged GT-assisted illustrative selection from the audited five-column figure",
        "prediction_geometry_modified": False,
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    caption = (
        "图X｜ScanNet和CA-1M中选定目标的三维定性比较。每行从左至右为GT、"
        "OnlineAnySeg、BoxFusion和CaRR-3D，并使用相同的世界坐标、场景网格、"
        "相机和视口。绿色虚线为GT，青色和蓝色分别为OnlineAnySeg与BoxFusion结果，"
        "橙色和金色分别为PTR与VAR恢复框。示例经过GT辅助选择，仅展示代表框；"
        "整体性能以定量实验为准。场景网格仅用于可视化。\n"
    )
    (OUT / "图注.txt").write_text(caption, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "exports": exports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
