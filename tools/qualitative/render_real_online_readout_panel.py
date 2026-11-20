#!/usr/bin/env python3
"""Render a compact real CaRR-3D readout for the framework figure.

The scene mesh, camera, and every displayed box come from the audited
scene0300_00 qualitative payload.  The selected output contains unchanged
native records plus measured PTR and VAR recoveries.  No detector is run and
no box geometry is edited.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.qualitative import add_onlineanyseg_four_scenes as source


OUT = ROOT / "figure_assets/online_detection_readout_real_scene0300_00_20261107"


def render(payload, renderer, *, transparent: bool, stem: str):
    # A portrait canvas matches the terminal readout panel in the method figure.
    fig = plt.figure(figsize=(3.15, 3.75), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1])
    source.draw_method(ax, payload, renderer, "ReCaR-3D", show_count=False)
    fig.savefig(OUT / f"{stem}.png", dpi=400, transparent=transparent,
                facecolor="none" if transparent else "white",
                bbox_inches="tight", pad_inches=.01)
    fig.savefig(OUT / f"{stem}.svg", transparent=transparent,
                facecolor="none" if transparent else "white",
                bbox_inches="tight", pad_inches=.01)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source.OUT = OUT
    source.opt.OUT = OUT
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

    renderer = source.original.load_renderer()
    payloads, audits, hashes = source.load_payloads(renderer)
    payload = next(p for p in payloads if p["spec"]["scene"] == "scene0300_00")
    audit = next(a for a in audits if a["scene"] == "scene0300_00")

    branches = [
        payload["predictions"]["ReCaR-3D"]["branches"][i]
        for i in payload["predictions"]["ReCaR-3D"]["selected"]
    ]
    assert "native" in branches and "plr" in branches and "calr" in branches

    render(payload, renderer, transparent=False,
           stem="online_detection_readout_real_white")
    render(payload, renderer, transparent=True,
           stem="online_detection_readout_real_transparent")

    manifest = {
        "schema": "carr3d.framework.real_online_readout.v1",
        "scene": "scene0300_00",
        "state": "terminal online state (t = T)",
        "displayed_prediction_indices": payload["predictions"]["ReCaR-3D"]["selected"],
        "displayed_branches": branches,
        "colors": {
            "native_hsr_scored_records": source.opt.COLORS["native"],
            "ptr_recovery": source.opt.COLORS["plr"],
            "var_recovery": source.opt.COLORS["calr"],
        },
        "prediction_geometry_modified": False,
        "selection": "Same audited GT-assisted representative selection used by the qualitative figure; not all predictions",
        "scene_audit": audit,
        "input_sha256": hashes,
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "使用说明.txt").write_text(
        "真实终态检测读出示例（scene0300_00）。蓝色为经HSR更新分数后的原生记录，"
        "橙色和金色分别为PTR与VAR恢复框；HSR不改变蓝色框的身份和几何。"
        "该图沿用论文定性图的GT辅助代表框选择，并非完整输出。所有框坐标均来自归档预测，"
        "场景网格仅用于可视化。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "branches": branches}, ensure_ascii=False))


if __name__ == "__main__":
    main()
