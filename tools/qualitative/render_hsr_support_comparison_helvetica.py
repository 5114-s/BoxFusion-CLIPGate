#!/usr/bin/env python3
"""Render the compact HSR support comparison with Helvetica typography."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "figure_assets/hsr_support_comparison_helvetica_20261109"

LABELS = (
    "Historical Max Support",
    "Last-Matched Support",
    "Current-Frame Support",
)
VALUES = (3.91, 1.29, 0.95)
COLORS = ("#326F9E", "#8799A8", "#A9B6C1")
INK = "#172A40"


def render() -> tuple[list[Path], Path]:
    fig = plt.figure(figsize=(8.40, 2.90), facecolor="white")
    ax = fig.add_axes([0.270, 0.205, 0.705, 0.745])

    y = (2, 1, 0)
    ax.barh(y, VALUES, height=0.47, color=COLORS, edgecolor="none", zorder=3)

    ax.set_xlim(0, 4.55)
    ax.set_ylim(-0.38, 2.38)
    ax.set_xticks((0, 1, 2, 3, 4))
    ax.set_yticks(y, LABELS)
    ax.set_xlabel(
        "AP25 Gain (percentage points)",
        fontsize=12.2,
        color=INK,
        labelpad=9,
    )
    ax.grid(axis="x", color="#D8E0E7", linewidth=0.9, zorder=0)

    ax.tick_params(axis="x", length=0, pad=7, labelsize=13.0, colors="#111111")
    ax.tick_params(axis="y", length=0, pad=16, labelsize=13.2, colors="#111111")
    for index, tick in enumerate(ax.get_yticklabels()):
        tick.set_fontweight("bold" if index == 0 else "normal")

    for ypos, value in zip(y, VALUES):
        ax.text(
            value + 0.11,
            ypos,
            f"+{value:.2f}",
            ha="left",
            va="center",
            fontsize=13.0,
            color=INK,
        )

    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9EADBA")
    ax.spines["bottom"].set_linewidth(0.9)

    outputs: list[Path] = []
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"hsr_support_comparison_helvetica.{suffix}"
        fig.savefig(path, dpi=300, facecolor="white")
        outputs.append(path)
    preview = OUT / "hsr_support_comparison_helvetica_preview.png"
    fig.savefig(preview, dpi=150, facecolor="white")
    plt.close(fig)

    svg_path = OUT / "hsr_support_comparison_helvetica.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_text = svg_text.replace("'Nimbus Sans'", "'Helvetica'")
    svg_text = svg_text.replace(
        "'Helvetica', 'Helvetica', sans-serif", "'Helvetica', sans-serif"
    )
    svg_path.write_text(svg_text, encoding="utf-8")
    return outputs, preview


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Nimbus Sans"],
        "pdf.use14corefonts": True,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    outputs, preview = render()
    print(f"PNG: {outputs[0]}")
    print(f"PDF: {outputs[1]}")
    print(f"SVG: {outputs[2]}")
    print(f"Preview: {preview}")


if __name__ == "__main__":
    main()
