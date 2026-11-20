#!/usr/bin/env python3
"""Render the final ScanNet MVSR support/correctness and ranking analysis."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


MAIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN))
from tools.audit_ca1m_nms_child_headroom import pairwise_iou
from tools.audit_final_ledger import scannet_inputs


ZERO_EPS = 1e-10
BIN_LABELS = (
    "0\n($S_i^{\\max}\\leq0.5$)",
    "(0, 0.1]",
    "(0.1, 0.2]",
    "(0.2, 0.3]",
    ">0.3",
)


def read_boxes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    boxes = np.asarray([row[1] for row in rows], dtype=np.float64).reshape(-1, 8, 3)
    scores = np.asarray([row[2] for row in rows], dtype=np.float64)
    return boxes, scores


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(values / (1.0 - values))


def ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    result = np.empty(len(scores), dtype=np.int64)
    result[order] = np.arange(1, len(scores) + 1)
    return result


def greedy_tp_labels(
    boxes: np.ndarray,
    scores: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float,
) -> np.ndarray:
    labels = np.zeros(len(boxes), dtype=bool)
    if not len(boxes) or not len(ground_truth):
        return labels
    overlaps = pairwise_iou(boxes, ground_truth)
    used: set[int] = set()
    for row in np.argsort(-scores, kind="stable"):
        choices = [
            (float(overlaps[row, column]), column)
            for column in range(len(ground_truth))
            if column not in used
        ]
        if not choices:
            continue
        value, column = max(choices)
        if value > threshold:
            labels[row] = True
            used.add(column)
    return labels


def bin_indices(effective: np.ndarray) -> np.ndarray:
    result = np.full(len(effective), -1, dtype=np.int64)
    result[effective <= ZERO_EPS] = 0
    result[(effective > ZERO_EPS) & (effective <= 0.1)] = 1
    result[(effective > 0.1) & (effective <= 0.2)] = 2
    result[(effective > 0.2) & (effective <= 0.3)] = 3
    result[effective > 0.3] = 4
    if np.any(result < 0):
        raise RuntimeError("effective-support bins do not partition all boxes")
    return result


def scene_bootstrap(
    counts: np.ndarray,
    tp_counts: np.ndarray,
    rank_sums: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scene_count = counts.shape[0]
    weights = rng.multinomial(
        scene_count,
        np.full(scene_count, 1.0 / scene_count),
        size=iterations,
    )
    sampled_counts = weights @ counts
    sampled_tp = weights @ tp_counts
    sampled_rank_sum = weights @ rank_sums
    with np.errstate(divide="ignore", invalid="ignore"):
        tp_rate = sampled_tp / sampled_counts
        rank_mean = sampled_rank_sum / sampled_counts
    tp_ci = np.nanpercentile(tp_rate, [2.5, 97.5], axis=0).T
    rank_ci = np.nanpercentile(rank_mean, [2.5, 97.5], axis=0).T
    return tp_ci, rank_ci


def group_bootstrap(
    counts: np.ndarray,
    effective_sums: np.ndarray,
    positive_counts: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap TP/FP support summaries with scenes as the sampling unit."""
    rng = np.random.default_rng(seed)
    scene_count = counts.shape[0]
    weights = rng.multinomial(
        scene_count,
        np.full(scene_count, 1.0 / scene_count),
        size=iterations,
    )
    sampled_counts = weights @ counts
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_effective = (weights @ effective_sums) / sampled_counts
        positive_fraction = (weights @ positive_counts) / sampled_counts
    return (
        np.nanpercentile(mean_effective, [2.5, 97.5], axis=0).T,
        np.nanpercentile(positive_fraction, [2.5, 97.5], axis=0).T,
    )


def analyze(
    scene_list: Path,
    components: Path,
    reranked_component: str,
    iou: float,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    expected_native_boxes: int | None = None,
) -> dict:
    scenes = [
        line.strip()
        for line in scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected the unique ScanNet official-100 scene list")

    ground_truth, alignments, transform = scannet_inputs()
    counts = np.zeros((len(scenes), 5), dtype=np.int64)
    tp_counts = np.zeros_like(counts)
    rank_sums = np.zeros((len(scenes), 5), dtype=np.float64)
    group_counts = np.zeros((len(scenes), 2), dtype=np.int64)
    group_effective_sums = np.zeros((len(scenes), 2), dtype=np.float64)
    group_positive_counts = np.zeros((len(scenes), 2), dtype=np.int64)

    for scene_index, scene in enumerate(scenes):
        filename = f"{scene}_boxes.pkl"
        boxes, original = read_boxes(components / "native_native" / filename)
        reranked_boxes, reranked = read_boxes(components / reranked_component / filename)
        if not np.array_equal(boxes, reranked_boxes):
            raise RuntimeError(f"MVSR changed native-box geometry: {scene}")
        if len(boxes) != len(reranked):
            raise RuntimeError(f"MVSR changed native-box count: {scene}")

        aligned = transform(boxes, alignments[scene]) if len(boxes) else boxes
        tp = greedy_tp_labels(aligned, original, ground_truth[scene], iou)
        rank_gain = ranks(original) - ranks(reranked)
        # The final native-logit update uses lambda=2 and tau_s=0.5, so the
        # effective support above the activation threshold can be recovered
        # exactly from the serialized before/after scores.
        effective = np.maximum(0.0, (logit(reranked) - logit(original)) / 2.0)
        bins = bin_indices(effective)
        for group_index, selected in enumerate((tp, ~tp)):
            group_counts[scene_index, group_index] = int(np.sum(selected))
            group_effective_sums[scene_index, group_index] = float(
                np.sum(effective[selected])
            )
            group_positive_counts[scene_index, group_index] = int(
                np.sum(effective[selected] > ZERO_EPS)
            )
        for bin_index in range(5):
            selected = bins == bin_index
            counts[scene_index, bin_index] = int(np.sum(selected))
            tp_counts[scene_index, bin_index] = int(np.sum(tp[selected]))
            rank_sums[scene_index, bin_index] = float(np.sum(rank_gain[selected]))

    total_counts = np.sum(counts, axis=0)
    total_tp = np.sum(tp_counts, axis=0)
    total_rank = np.sum(rank_sums, axis=0)
    native_box_count = int(np.sum(total_counts))
    if expected_native_boxes is not None and native_box_count != expected_native_boxes:
        raise RuntimeError(
            f"expected {expected_native_boxes:,} final native boxes, "
            f"found {native_box_count:,}"
        )
    tp_rate = total_tp / total_counts
    rank_mean = total_rank / total_counts
    tp_ci, rank_ci = scene_bootstrap(
        counts,
        tp_counts,
        rank_sums,
        bootstrap_iterations,
        bootstrap_seed,
    )
    total_group_counts = np.sum(group_counts, axis=0)
    total_group_effective = np.sum(group_effective_sums, axis=0)
    total_group_positive = np.sum(group_positive_counts, axis=0)
    group_effective_ci, group_positive_ci = group_bootstrap(
        group_counts,
        group_effective_sums,
        group_positive_counts,
        bootstrap_iterations,
        bootstrap_seed,
    )

    return {
        "schema": "boxfusion.scannet100.mvsr_support_ranking.no_visibility.v2",
        "scene_count": len(scenes),
        "native_box_count": native_box_count,
        "iou_threshold": iou,
        "tp_definition": "greedy one-to-one matching ordered by original native scores",
        "reranked_component": reranked_component,
        "mvsr_factors": ["proposal_score", "2d_overlap", "3d_consistency"],
        "temporal_aggregation": "historical_maximum",
        "effective_support": "Delta_i=max(0,S_i^max-tau_s), tau_s=0.5",
        "rank_scope": "within-scene native boxes only",
        "bootstrap": {
            "unit": "scene",
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "interval": "percentile_95",
        },
        "tp_fp_summary": {
            label: {
                "count": int(total_group_counts[index]),
                "mean_effective_support": float(
                    total_group_effective[index] / total_group_counts[index]
                ),
                "mean_effective_support_ci95": [
                    float(value) for value in group_effective_ci[index]
                ],
                "positive_correction_fraction": float(
                    total_group_positive[index] / total_group_counts[index]
                ),
                "positive_correction_fraction_ci95": [
                    float(value) for value in group_positive_ci[index]
                ],
                # Scores serialize max(0, S_i^max-tau_s), so the raw S_i^max
                # is exactly identifiable only for positively corrected rows.
                "mean_composite_support_among_positive_corrections": (
                    float(
                        0.5
                        + total_group_effective[index]
                        / total_group_positive[index]
                    )
                    if total_group_positive[index]
                    else None
                ),
                "all_rows_mean_composite_support_identifiability_bounds": [
                    float(
                        total_group_effective[index] / total_group_counts[index]
                        + 0.5
                        * total_group_positive[index] / total_group_counts[index]
                    ),
                    float(
                        total_group_effective[index] / total_group_counts[index]
                        + 0.5
                    ),
                ],
            }
            for index, label in enumerate(("TP", "FP"))
        },
        "bins": [
            {
                "label": label,
                "count": int(total_counts[index]),
                "tp_rate": float(tp_rate[index]),
                "tp_rate_ci95": [float(value) for value in tp_ci[index]],
                "mean_rank_change": float(rank_mean[index]),
                "mean_rank_change_ci95": [float(value) for value in rank_ci[index]],
            }
            for index, label in enumerate(BIN_LABELS)
        ],
    }


def render(result: dict, output_dir: Path) -> None:
    bins = result["bins"]
    counts = np.asarray([row["count"] for row in bins])
    tp_rate = np.asarray([row["tp_rate"] for row in bins])
    tp_ci = np.asarray([row["tp_rate_ci95"] for row in bins])
    rank_mean = np.asarray([row["mean_rank_change"] for row in bins])
    rank_ci = np.asarray([row["mean_rank_change_ci95"] for row in bins])
    x = np.arange(5)

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
        figsize=(7.20, 2.95),
        gridspec_kw={"left": 0.078, "right": 0.992, "bottom": 0.245, "top": 0.79, "wspace": 0.30},
    )

    blue = ("#D6E3F1", "#B6D0E6", "#8EB8D7", "#5D9CC7", "#327BAE")
    rank_colors = ("#9EA5AF", "#B8BEC7", "#A49BD0", "#7F73BB", "#5F63A9")

    left_yerr = np.vstack((tp_rate - tp_ci[:, 0], tp_ci[:, 1] - tp_rate))
    left.bar(x, tp_rate, color=blue, width=0.68, edgecolor="white", linewidth=0.7, zorder=3)
    left.errorbar(
        x,
        tp_rate,
        yerr=left_yerr,
        fmt="none",
        ecolor="#3F444A",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        zorder=5,
    )
    for index, (rate, count, upper) in enumerate(zip(tp_rate, counts, tp_ci[:, 1])):
        text_color = "white" if index >= 3 else "#30343A"
        left.text(x[index], rate * 0.52, f"{100.0 * rate:.1f}%", ha="center", va="center", fontsize=7.0, color=text_color)
        left.text(x[index], upper + 0.025, f"n={count}", ha="center", va="bottom", fontsize=6.6, color="#555B61")

    left.set_ylim(0, 0.90)
    left.set_yticks(np.arange(0, 0.91, 0.2))
    left.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    left.set_ylabel("TP rate")
    left.set_title("Higher support correlates with native-box correctness", pad=18)
    left.text(
        0.5,
        1.035,
        f"Greedy matching at 3D AABB IoU threshold 0.15 "
        f"(n={result['native_box_count']:,})",
        transform=left.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#555B61",
    )

    right_yerr = np.vstack((rank_mean - rank_ci[:, 0], rank_ci[:, 1] - rank_mean))
    right.bar(x, rank_mean, color=rank_colors, width=0.68, edgecolor="white", linewidth=0.7, zorder=3)
    right.errorbar(
        x,
        rank_mean,
        yerr=right_yerr,
        fmt="none",
        ecolor="#3F444A",
        elinewidth=0.8,
        capsize=2.2,
        capthick=0.8,
        zorder=5,
    )
    for index, (value, interval) in enumerate(zip(rank_mean, rank_ci)):
        if value >= 0:
            y, va = interval[1] + 0.12, "bottom"
        else:
            y, va = interval[0] - 0.12, "top"
        right.text(x[index], y, f"{value:+.2f}", ha="center", va=va, fontsize=6.9, color="#41464C")

    right.axhline(0, color="#30343A", linewidth=0.9, zorder=4)
    right.set_ylim(-3.35, 4.05)
    right.set_yticks(np.arange(-3, 4.1, 1))
    right.set_ylabel("Mean within-scene native-box rank change\n(positive = higher rank)")
    right.set_title("High-support native boxes move upward", pad=18)
    right.text(
        0.5,
        1.035,
        "Fixed native-box set; only confidence scores change",
        transform=right.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#555B61",
    )

    for axis, panel in ((left, "a"), (right, "b")):
        axis.set_xticks(x, BIN_LABELS)
        axis.set_xlabel(r"Effective-support bin $\Delta_i$")
        axis.grid(axis="y", color="#E4E7EA", linewidth=0.6, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)
        axis.tick_params(axis="both", labelsize=7.0, width=0.7, length=3)
        axis.text(-0.14, 1.095, panel, transform=axis.transAxes, fontsize=9.2, fontweight="bold", va="top")

    fig.text(
        0.535,
        0.045,
        r"$\Delta_i=\max(0,S_i^{\max}-\tau_s)$, $\tau_s=0.5$; error bars: scene-bootstrap 95% CI",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#555B61",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "mvsr_support_ranking_optimized"
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_report(result: dict, output_dir: Path) -> None:
    """Write manuscript-ready tables without relabelling effective support."""
    lines = [
        "# Final ScanNet-100 MVSR support and ranking statistics",
        "",
        f"Native boxes: {result['native_box_count']:,}; IoU={result['iou_threshold']:.2f}; "
        "three-factor support without visibility; historical-maximum aggregation.",
        "",
        "The serialized score update identifies the effective support "
        r"$\Delta_i=\max(0,S_i^{\max}-0.5)$ exactly. For rows with "
        r"$\Delta_i=0$, the raw composite support is only known to satisfy "
        r"$S_i^{\max}\leq0.5$; therefore the report does not invent an exact "
        "all-row mean raw support.",
        "",
        "## Support bins",
        "",
        "| Effective-support bin | n | TP rate | TP-rate 95% CI | Mean native-box rank change | Rank-change 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["bins"]:
        lines.append(
            f"| {row['label'].replace(chr(10), ' ')} | {row['count']:,} | "
            f"{100.0 * row['tp_rate']:.2f}% | "
            f"[{100.0 * row['tp_rate_ci95'][0]:.2f}%, {100.0 * row['tp_rate_ci95'][1]:.2f}%] | "
            f"{row['mean_rank_change']:+.3f} | "
            f"[{row['mean_rank_change_ci95'][0]:+.3f}, {row['mean_rank_change_ci95'][1]:+.3f}] |"
        )
    lines.extend([
        "",
        "## TP/FP summary",
        "",
        "| Group | n | Mean effective support | Positive correction | Mean raw composite support among corrected rows | All-row raw-support bounds |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label in ("TP", "FP"):
        row = result["tp_fp_summary"][label]
        positive_mean = row["mean_composite_support_among_positive_corrections"]
        bounds = row["all_rows_mean_composite_support_identifiability_bounds"]
        lines.append(
            f"| {label} | {row['count']:,} | {row['mean_effective_support']:.4f} | "
            f"{100.0 * row['positive_correction_fraction']:.2f}% | "
            f"{positive_mean:.4f} | [{bounds[0]:.4f}, {bounds[1]:.4f}] |"
        )
    lines.extend([
        "",
        "Positive rank change means the native box moves upward within the fixed "
        "per-scene native-box list. TP labels use greedy one-to-one matching in "
        "the original-score order at 3D AABB IoU 0.15. Error bars are paired "
        "scene-bootstrap percentile 95% intervals.",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=MAIN / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument(
        "--components",
        type=Path,
        default=MAIN / "development/final_controls/experiments/mvsr_design_20261005/components",
    )
    parser.add_argument("--reranked-component", default="native_factor_no_visibility")
    parser.add_argument("--iou", type=float, default=0.15)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261006)
    parser.add_argument("--expected-native-boxes", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MAIN / "figure_assets/mvsr_support_ranking",
    )
    args = parser.parse_args()
    result = analyze(
        args.scene_list,
        args.components,
        args.reranked_component,
        args.iou,
        args.bootstrap_iterations,
        args.bootstrap_seed,
        args.expected_native_boxes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "mvsr_support_ranking_optimized.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    render(result, args.output_dir)
    write_report(result, args.output_dir)


if __name__ == "__main__":
    main()
