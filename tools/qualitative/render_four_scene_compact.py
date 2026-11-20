#!/usr/bin/env python3
"""Compact, geometry-preserving four-scene qualitative comparison.

This renderer reuses the audited payloads, cameras, selections, colors, and
draw functions from ``render_four_scene_without_boxfusionstar.py``.  It changes
only figure spacing and omits the separate legend/footer so the paper panel can
use more of its canvas for the four-by-four comparison grid.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.qualitative import add_onlineanyseg_four_scenes as source


OUT = ROOT / "figure_assets/qualitative_scannet_ca1m_onlineanyseg_four_columns_compact_20261109"
METHODS = ("Ground truth", "OnlineAnySeg", "BoxFusion", "ReCaR-3D")
TITLES = ("GT", "OnlineAnySeg", "BoxFusion", "CaRR-3D (Ours)")


def render(payloads, renderer):
    rows = len(payloads)
    # The former export was 5.82 in tall.  A 5.05 in canvas plus small negative
    # inter-panel spacing removes empty gutters without cropping any axes.
    height = 5.05
    fig, axes = plt.subplots(
        rows,
        len(METHODS),
        figsize=(source.opt.WIDTH, height),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.046,
        right=0.998,
        bottom=0.006,
        top=0.936,
        wspace=-0.025,
        hspace=-0.105,
    )

    for column, title in enumerate(TITLES):
        bounds = axes[0, column].get_position()
        fig.text(
            bounds.x0 + bounds.width / 2,
            0.978,
            title,
            ha="center",
            va="center",
            fontsize=8.0,
            fontweight="bold",
            color=source.opt.INK,
        )

    for row, payload in enumerate(payloads):
        for column, method in enumerate(METHODS):
            source.draw_method(
                axes[row, column], payload, renderer, method, show_count=False
            )

        bounds = axes[row, 0].get_position()
        dataset = payload["spec"].get("dataset", "ScanNet")
        fig.text(
            0.014,
            bounds.y0 + bounds.height / 2,
            f"({chr(97 + row)})  {dataset}\n{payload['spec']['scene']}",
            rotation=90,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color=source.opt.INK,
            linespacing=1.15,
        )

    paths = []
    stem = OUT / "qualitative_3d_gt_onlineanyseg_boxfusion_carr_four_scenes_compact"
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=400 if suffix == "png" else None, facecolor="white")
        paths.append(path.name)
    preview = stem.with_name(stem.name + "_preview").with_suffix(".png")
    fig.savefig(preview, dpi=170, facecolor="white")
    plt.close(fig)
    return paths + [preview.name]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source.OUT = OUT
    source.opt.OUT = OUT
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "text.color": source.opt.INK,
        }
    )
    renderer = source.original.load_renderer()
    payloads, audits, hashes = source.load_payloads(renderer)
    exports = render(payloads, renderer)
    manifest = {
        "schema": "carr3d.qualitative.four_columns.compact.v1",
        "layout_only_change": True,
        "methods": list(TITLES),
        "scenes": audits,
        "input_sha256": hashes,
        "exports": exports,
        "prediction_geometry_modified": False,
        "spacing": {
            "figure_height_inches": 5.05,
            "wspace": -0.025,
            "hspace": -0.105,
            "legend_and_footer_omitted": True,
        },
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(OUT), "exports": exports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
