#!/usr/bin/env python3
"""Render the highest-IoU audited recovery as a compact readout panel."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.qualitative import add_onlineanyseg_four_scenes as source


OUT = ROOT / "figure_assets/online_detection_readout_real_high_iou_20261107"
SCENE = "42897692"


def export(payload, renderer, *, transparent: bool, stem: str):
    fig = plt.figure(figsize=(3.6, 3.0), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1])
    source.draw_method(ax, payload, renderer, "ReCaR-3D", show_count=False)
    kwargs = {
        "dpi": 400,
        "transparent": transparent,
        "facecolor": "none" if transparent else "white",
        "bbox_inches": "tight",
        "pad_inches": .01,
    }
    fig.savefig(OUT / f"{stem}.png", **kwargs)
    kwargs.pop("dpi")
    fig.savefig(OUT / f"{stem}.svg", **kwargs)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source.OUT = OUT
    source.opt.OUT = OUT
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

    renderer = source.original.load_renderer()
    payloads, audits, hashes = source.load_payloads(renderer)
    payload = next(p for p in payloads if p["spec"]["scene"] == SCENE)
    audit = next(a for a in audits if a["scene"] == SCENE)
    recovery = payload["recoveries"][0]
    assert recovery["branch"] == "VAR"
    assert recovery["recovered_iou"] > .90

    export(payload, renderer, transparent=False,
           stem="online_detection_readout_high_iou_white")
    export(payload, renderer, transparent=True,
           stem="online_detection_readout_high_iou_transparent")

    selected = payload["predictions"]["ReCaR-3D"]["selected"]
    branches = [payload["predictions"]["ReCaR-3D"]["branches"][i] for i in selected]
    manifest = {
        "schema": "carr3d.framework.real_online_readout.high_iou.v1",
        "scene": SCENE,
        "dataset": "CA-1M",
        "state": "terminal online state (t = T)",
        "displayed_prediction_indices": selected,
        "displayed_branches": branches,
        "recovery": recovery,
        "recovered_3d_aabb_iou": recovery["recovered_iou"],
        "prediction_geometry_modified": False,
        "selection": "Same audited GT-assisted representative selection used by the qualitative figure; not all predictions",
        "scene_audit": audit,
        "input_sha256": hashes,
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "使用说明.txt").write_text(
        f"真实终态检测读出示例（CA-1M {SCENE}）。蓝色为原生记录，金色为VAR恢复框；"
        f"恢复框与对应GT的三维AABB IoU为{recovery['recovered_iou']:.3f}。"
        "所有框坐标均来自归档预测，未修改几何；场景网格仅用于可视化。\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "iou": recovery["recovered_iou"],
                      "branches": branches}, ensure_ascii=False))


if __name__ == "__main__":
    main()
