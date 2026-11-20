#!/usr/bin/env python3
"""Render the paper figure for candidate evidence and recovery attribution."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


TOTAL_GT = 1433
IOU_LABELS = ("0.15", "0.25", "0.50")

# Panel a: mutually exclusive categories. Targets with both proposal and anchor
# evidence are assigned to Proposal evidence; Anchor-only therefore means no
# post-NMS proposal evidence was available.
EVIDENCE_COUNTS = {
    "Native map": np.array([908, 825, 551]),
    "Proposal evidence": np.array([431, 437, 397]),
    "Anchor-only evidence": np.array([54, 71, 103]),
    "No evidence": np.array([40, 100, 382]),
}

# Panel b: mutually exclusive recovery attribution among native-missed GT.
MISSED_COUNTS = np.array([525, 608, 882])
RECOVERY_COUNTS = {
    "PLR only": np.array([6, 8, 11]),
    "CALR only": np.array([196, 197, 197]),
    "PLR + CALR": np.array([187, 181, 81]),
    "Not recovered": np.array([136, 222, 593]),
}

COLORS = {
    "Native map": "#4E79A7",
    "Proposal evidence": "#F28E2B",
    "Anchor-only evidence": "#DDB83B",
    "No evidence": "#C7CCD2",
    "PLR only": "#F28E2B",
    "CALR only": "#DDB83B",
    "PLR + CALR": "#8B6BBE",
    "Not recovered": "#C7CCD2",
}


def _style_axis(axis: mpl.axes.Axes) -> None:
    axis.set_ylim(0, 1)
    axis.set_yticks(np.linspace(0, 1, 6))
    axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axis.grid(axis="y", color="#E6E8EB", linewidth=0.6, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", labelsize=7.2, width=0.7, length=3)


def _label_regular_segments(
    axis: mpl.axes.Axes,
    x: np.ndarray,
    bottoms: np.ndarray,
    counts: np.ndarray,
    denominators: np.ndarray,
    *,
    text_color: str,
    minimum_fraction: float,
) -> None:
    fractions = counts / denominators
    for index, (count, fraction) in enumerate(zip(counts, fractions)):
        if fraction < minimum_fraction:
            continue
        axis.text(
            x[index],
            bottoms[index] + fraction / 2,
            f"{int(count)}",
            ha="center",
            va="center",
            fontsize=7.1,
            color=text_color,
            zorder=5,
        )


def render(output_dir: Path) -> None:
    for values in EVIDENCE_COUNTS.values():
        if values.shape != (3,):
            raise RuntimeError("invalid evidence count shape")
    if not np.all(sum(EVIDENCE_COUNTS.values()) == TOTAL_GT):
        raise RuntimeError("panel-a counts do not sum to 1,433")
    if not np.all(sum(RECOVERY_COUNTS.values()) == MISSED_COUNTS):
        raise RuntimeError("panel-b counts do not match native-missed denominators")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(7.20, 3.02),
        gridspec_kw={"wspace": 0.28, "left": 0.075, "right": 0.992, "bottom": 0.18, "top": 0.73},
    )
    x = np.arange(3, dtype=float)
    width = 0.66

    # Panel a.
    _style_axis(left)
    left_bottom = np.zeros(3)
    left_denominator = np.full(3, TOTAL_GT)
    for name in EVIDENCE_COUNTS:
        counts = EVIDENCE_COUNTS[name]
        fraction = counts / TOTAL_GT
        left.bar(
            x,
            fraction,
            width,
            bottom=left_bottom,
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
        )
        text_color = "white" if name in ("Native map", "Proposal evidence") else "#333333"
        _label_regular_segments(
            left,
            x,
            left_bottom,
            counts,
            left_denominator,
            text_color=text_color,
            minimum_fraction=0.055,
        )
        left_bottom += fraction

    # External callouts for the two very thin segments in the first bar and the
    # missing anchor count in the second bar.
    left.annotate(
        "54",
        xy=(x[0] + width / 2, (908 + 431 + 27) / TOTAL_GT),
        xytext=(x[0] + 0.43, 0.925),
        ha="left",
        va="center",
        fontsize=7.0,
        color="#6F5A00",
        arrowprops={"arrowstyle": "-", "color": COLORS["Anchor-only evidence"], "lw": 0.8},
    )
    left.annotate(
        "40",
        xy=(x[0] + width / 2, (TOTAL_GT - 20) / TOTAL_GT),
        xytext=(x[0] + 0.43, 0.992),
        ha="left",
        va="center",
        fontsize=7.0,
        color="#42464B",
        arrowprops={"arrowstyle": "-", "color": "#8D949B", "lw": 0.8},
    )
    left.annotate(
        "71",
        xy=(x[1] + width / 2, (825 + 437 + 35.5) / TOTAL_GT),
        xytext=(x[1] + 0.43, 0.905),
        ha="left",
        va="center",
        fontsize=7.0,
        color="#6F5A00",
        arrowprops={"arrowstyle": "-", "color": COLORS["Anchor-only evidence"], "lw": 0.8},
    )

    left.set_xticks(x, IOU_LABELS)
    left.set_xlabel("3D AABB IoU threshold")
    left.set_ylabel("Fraction of all GT objects")
    left.set_title("Evidence remains beyond the native map", pad=20)
    left.text(
        0.5,
        1.055,
        "All GT objects (n = 1,433)",
        transform=left.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#555B61",
    )
    left.text(-0.13, 1.095, "a", transform=left.transAxes, fontsize=9.2, fontweight="bold", va="top")

    # Panel b.
    _style_axis(right)
    right_bottom = np.zeros(3)
    for name in RECOVERY_COUNTS:
        counts = RECOVERY_COUNTS[name]
        fraction = counts / MISSED_COUNTS
        right.bar(
            x,
            fraction,
            width,
            bottom=right_bottom,
            color=COLORS[name],
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
        )
        if name != "PLR only":
            text_color = "white" if name == "PLR + CALR" else "#333333"
            _label_regular_segments(
                right,
                x,
                right_bottom,
                counts,
                MISSED_COUNTS,
                text_color=text_color,
                minimum_fraction=0.045,
            )
        right_bottom += fraction

    # External callouts preserve the true 1% PLR-only slivers without visually
    # inflating them.
    for index, count in enumerate(RECOVERY_COUNTS["PLR only"]):
        fraction = count / MISSED_COUNTS[index]
        right.annotate(
            f"{int(count)}",
            xy=(x[index], fraction / 2),
            xytext=(x[index], 0.055),
            ha="center",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=COLORS["PLR only"],
            bbox={"boxstyle": "round,pad=0.08", "fc": "white", "ec": "none", "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": COLORS["PLR only"], "lw": 0.75},
            zorder=6,
        )

    right.set_xticks(x, [f"{iou}\n(n={denom})" for iou, denom in zip(IOU_LABELS, MISSED_COUNTS)])
    right.set_xlabel("3D AABB IoU threshold (native-missed GT)")
    right.set_ylabel("Fraction of native-missed GT objects")
    right.set_title("CALR contributes to most recovered targets", pad=20)
    right.text(
        0.5,
        1.055,
        "Denominator varies with IoU threshold",
        transform=right.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#555B61",
    )
    right.text(-0.13, 1.095, "b", transform=right.transAxes, fontsize=9.2, fontweight="bold", va="top")

    left_handles = [Patch(facecolor=COLORS[name], edgecolor="white", label=name) for name in EVIDENCE_COUNTS]
    right_handles = [Patch(facecolor=COLORS[name], edgecolor="white", label=name) for name in RECOVERY_COUNTS]
    left.legend(
        handles=left_handles,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.275),
        frameon=False,
        fontsize=7.0,
        handlelength=1.5,
        columnspacing=1.6,
        handletextpad=0.45,
        borderaxespad=0,
    )
    right.legend(
        handles=right_handles,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.275),
        frameon=False,
        fontsize=7.0,
        handlelength=1.5,
        columnspacing=1.6,
        handletextpad=0.45,
        borderaxespad=0,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "candidate_evidence_recovery_optimized"
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figure_assets/candidate_evidence_recovery"),
    )
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
