#!/usr/bin/env python3
"""Render the paper-ready 2x2 qualitative evidence figure."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

COLORS = {
    "native": "#3977B7",
    "plr": "#E67822",
    "calr": "#B77900",
    "gt": "#159B62",
    "failure": "#D62728",
    "support": "#8055B7",
}
TITLES = {
    "plr_recovery": "(a) PLR candidate recovery",
    "calr_recovery": "(b) CALR candidate recovery",
    "mvsr_reranking": "(c) HCSR native-record reranking",
    "failure": "(d) High-support false positive",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NO_VISIBILITY_COMPONENTS = (
    REPO_ROOT
    / "development/final_controls/experiments/mvsr_design_20261005/components"
    / "native_factor_no_visibility"
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    labels = np.asarray([row[0] for row in rows], dtype=np.int64)
    boxes = np.asarray([row[1] for row in rows], dtype=np.float64).reshape(-1, 8, 3)
    scores = np.asarray([row[2] for row in rows], dtype=np.float64)
    return labels, boxes, scores


def read_boxes(path: Path) -> np.ndarray:
    return read_predictions(path)[1]


def stable_rank(scores: np.ndarray, index: int) -> int:
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    return int(np.flatnonzero(order == int(index))[0]) + 1


def logit(value: float) -> float:
    clipped = float(np.clip(value, 1.0e-8, 1.0 - 1.0e-8))
    return math.log(clipped / (1.0 - clipped))


def final_mvsr_metrics(case, native_root: Path, reranked_root: Path) -> dict:
    """Resolve the selected box in the final no-visibility MVSR output."""
    scene = case["scene"]
    native_path = native_root / f"{scene}_boxes.pkl"
    reranked_path = reranked_root / f"{scene}_boxes.pkl"
    native_labels, native_boxes, native_scores = read_predictions(native_path)
    reranked_labels, reranked_boxes, reranked_scores = read_predictions(reranked_path)
    if not np.array_equal(native_labels, reranked_labels):
        raise RuntimeError(f"MVSR changed native labels: {scene}")
    if not np.array_equal(native_boxes, reranked_boxes):
        raise RuntimeError(f"MVSR changed native geometry: {scene}")

    target = np.asarray(case["box"], dtype=np.float64).reshape(8, 3)
    distances = np.max(np.abs(native_boxes - target[None]), axis=(1, 2))
    index = int(np.argmin(distances))
    if float(distances[index]) > 1.0e-4:
        raise RuntimeError(
            f"selected native box absent from final MVSR components: "
            f"{scene}, max corner error={distances[index]:.6g}"
        )

    original_score = float(native_scores[index])
    reranked_score = float(reranked_scores[index])
    # Final MVSR uses logit_weight=2 and support_threshold=0.5. For an
    # activated update, invert the serialized score transform to recover the
    # historical maximum of the three-factor composite support.
    support_delta = max(0.0, (logit(reranked_score) - logit(original_score)) / 2.0)
    if support_delta <= 0.0:
        raise RuntimeError(f"selected MVSR case is not promoted in final route: {scene}")
    max_support = 0.5 + support_delta
    return {
        "scene": scene,
        "native_index": index,
        "original_score": original_score,
        "reranked_score": reranked_score,
        "original_rank": stable_rank(native_scores, index),
        "reranked_rank": stable_rank(reranked_scores, index),
        "max_three_factor_support": max_support,
        "native_component": str(native_path.resolve()),
        "native_sha256": digest(native_path),
        "reranked_component": str(reranked_path.resolve()),
        "reranked_sha256": digest(reranked_path),
    }


def alignment(raw_root: Path, scene: str) -> np.ndarray:
    path = raw_root / scene / f"{scene}.txt"
    for line in path.read_text().splitlines():
        if line.startswith("axisAlignment ="):
            return np.asarray(line.split("=", 1)[1].split(), dtype=float).reshape(4, 4)
    raise RuntimeError(f"missing axisAlignment: {path}")


def align_boxes(boxes: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.einsum("ij,nkj->nki", matrix[:3, :3], boxes) + matrix[:3, 3]


def project_box(
    box: np.ndarray,
    raw_root: Path,
    scene: str,
    frame: int,
    output_size=(640, 480),
) -> list[float] | None:
    pose = np.loadtxt(raw_root / scene / "pose" / f"{frame}.txt")
    intrinsic = np.loadtxt(
        raw_root / scene / "intrinsic" / "intrinsic_color.txt"
    )[:3, :3]
    world = np.c_[np.asarray(box, dtype=float).reshape(8, 3), np.ones(8)]
    camera = (np.linalg.inv(pose) @ world.T).T[:, :3]
    valid = camera[:, 2] > 1e-4
    if valid.sum() < 2:
        return None
    uvw = (intrinsic @ camera[valid].T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    image = Image.open(raw_root / scene / "color" / f"{frame}.jpg")
    sx, sy = output_size[0] / image.width, output_size[1] / image.height
    uv[:, 0] *= sx
    uv[:, 1] *= sy
    lo, hi = uv.min(0), uv.max(0)
    lo = np.maximum(lo, [0, 0])
    hi = np.minimum(hi, [output_size[0] - 1, output_size[1] - 1])
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def draw_rect(ax, box, color, dashed=False, width=2.0, outlined=False):
    if box is None:
        return
    x0, y0, x1, y1 = box
    patch = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        edgecolor=color,
        linewidth=width,
        linestyle="--" if dashed else "-",
    )
    if outlined:
        patch.set_path_effects(
            [
                path_effects.Stroke(linewidth=width + 1.8, foreground="white"),
                path_effects.Normal(),
            ]
        )
    ax.add_patch(patch)


def crop_evidence_frame(image, focus_box, boxes, scale=1.45):
    """Crop around a 2D box while keeping the target at a consistent relative scale."""
    if focus_box is None:
        return image, boxes
    x0, y0, x1, y1 = [float(value) for value in focus_box]
    width = max(x1 - x0, 12.0)
    height = max(y1 - y0, 12.0)
    crop_width = max(width * scale, 160.0)
    crop_width = min(crop_width, 640.0)
    crop_height = min(crop_width * 3.0 / 4.0, 480.0)
    crop_width = crop_height * 4.0 / 3.0
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    left = min(max(center_x - crop_width / 2.0, 0.0), 640.0 - crop_width)
    top = min(max(center_y - crop_height / 2.0, 0.0), 480.0 - crop_height)
    right, bottom = left + crop_width, top + crop_height
    cropped = image.crop((left, top, right, bottom)).resize(
        (640, 480), Image.Resampling.LANCZOS
    )
    scale_x, scale_y = 640.0 / crop_width, 480.0 / crop_height

    transformed = {}
    for name, box in boxes.items():
        if box is None:
            transformed[name] = None
            continue
        bx0, by0, bx1, by1 = [float(value) for value in box]
        transformed[name] = [
            np.clip((bx0 - left) * scale_x, 0.0, 639.0),
            np.clip((by0 - top) * scale_y, 0.0, 479.0),
            np.clip((bx1 - left) * scale_x, 0.0, 639.0),
            np.clip((by1 - top) * scale_y, 0.0, 479.0),
        ]
    return cropped, transformed


def rgb_strip(
    ax,
    case,
    evidence,
    raw_root: Path,
    failure=False,
    slots=None,
    card_text=None,
    max_frames=None,
):
    rows = evidence["frames"][: int(max_frames or 3)]
    if not rows:
        raise RuntimeError(f"no RGB evidence for {case['scene']}")
    slots = max(len(rows), int(slots or len(rows)))
    canvas = np.full((480, 640 * slots, 3), 248, dtype=np.uint8)
    spans = []
    target = np.asarray(case["box"], dtype=float)

    for index, row in enumerate(rows):
        frame = int(row["frame_id"])
        image = (
            Image.open(row["rgb_path"])
            .convert("RGB")
            .resize((640, 480), Image.Resampling.LANCZOS)
        )
        candidate = row.get("candidate_xyxy")
        if candidate is None:
            candidate = project_box(
                row["candidate_box"], raw_root, case["scene"], frame
            )
        native = (
            project_box(target, raw_root, case["scene"], frame)
            if case["source"] == "native"
            else None
        )

        boxes = {"candidate": candidate, "native": native}
        if case["source"] == "plr":
            image, boxes = crop_evidence_frame(
                image, candidate, boxes, scale=1.45
            )
        elif case["source"] == "native":
            # Keep the native projection and its matched proposal legible at
            # double-column size.
            focus = candidate
            if candidate is not None and native is not None:
                focus = [
                    min(candidate[0], native[0]),
                    min(candidate[1], native[1]),
                    max(candidate[2], native[2]),
                    max(candidate[3], native[3]),
                ]
            image, boxes = crop_evidence_frame(
                image, focus, boxes, scale=1.85
            )

        offset = index * 640
        canvas[:, offset : offset + 640] = np.asarray(image)
        spans.append((offset, frame, boxes["candidate"], boxes["native"]))

    ax.imshow(canvas, aspect="auto")
    for xoff, frame, candidate, native in spans:
        candidate = (
            None
            if candidate is None
            else [
                candidate[0] + xoff,
                candidate[1],
                candidate[2] + xoff,
                candidate[3],
            ]
        )
        native = (
            None
            if native is None
            else [
                native[0] + xoff,
                native[1],
                native[2] + xoff,
                native[3],
            ]
        )
        if case["source"] == "native":
            draw_rect(
                ax,
                native,
                COLORS["failure"] if failure else COLORS["native"],
                width=3.2,
            )
            draw_rect(
                ax,
                candidate,
                COLORS["support"],
                dashed=True,
                width=1.7,
            )
        else:
            color = COLORS["failure"] if failure else COLORS[case["source"]]
            draw_rect(
                ax,
                candidate,
                color,
                width=2.1,
                outlined=case["source"] == "calr",
            )
        ax.text(
            xoff + 8,
            22,
            f"t={frame}",
            color="white",
            fontsize=6.7,
            bbox={
                "facecolor": "black",
                "alpha": 0.58,
                "edgecolor": "none",
                "pad": 1,
            },
        )

    if card_text:
        card_left = 640 * (slots - 1)
        ax.add_patch(
            Rectangle(
                (card_left + 24, 30),
                592,
                420,
                facecolor="#F7F8FA",
                edgecolor="#CBD5E1",
                linewidth=0.8,
            )
        )
        ax.text(
            card_left + 320,
            240,
            card_text,
            ha="center",
            va="center",
            fontsize=7.1,
            linespacing=1.45,
            color="#172033",
        )

    ax.set_xlim(0, 640 * slots)
    ax.set_ylim(480, 0)
    ax.axis("off")


def box2d(
    ax,
    corners,
    color,
    dashed=False,
    width=1.5,
    alpha=1.0,
    outlined=False,
):
    corners = np.asarray(corners, dtype=float)
    lo, hi = corners[:, :2].min(0), corners[:, :2].max(0)
    patch = Rectangle(
        lo,
        *(hi - lo),
        fill=False,
        edgecolor=color,
        linewidth=width,
        linestyle="--" if dashed else "-",
        alpha=alpha,
    )
    if outlined:
        patch.set_path_effects(
            [
                path_effects.Stroke(linewidth=width + 1.6, foreground="white"),
                path_effects.Normal(),
            ]
        )
    ax.add_patch(patch)


def topdown(
    ax,
    case,
    components: Path,
    gt_root: Path,
    raw_root: Path,
    failure=False,
    show_selected=True,
    crop_scale=1.25,
    min_margin=0.55,
    point_darken=0.72,
    point_alpha=0.52,
    point_size=0.28,
    monochrome_points=False,
    background_color="#F8FAFC",
):
    scene = case["scene"]
    vertices = np.load(gt_root / f"{scene}_vert.npy")
    if len(vertices) > 35000:
        vertices = vertices[:: max(1, len(vertices) // 35000)]
    if monochrome_points:
        # A neutral, higher-contrast rendering is easier to read for sparse
        # failure regions while preserving the measured scene geometry.
        colors = "#475569"
    elif vertices.shape[1] >= 6:
        # Darken light ScanNet colors so walls and pale surfaces remain visible
        # after reduction on a white paper background.
        colors = np.clip(
            (vertices[:, 3:6] / 255.0) * point_darken, 0.0, 1.0
        )
    else:
        colors = "#6B7280"
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        c=colors,
        s=point_size,
        alpha=point_alpha,
        linewidth=0,
        rasterized=True,
    )

    matrix = alignment(raw_root, scene)
    native = align_boxes(
        read_boxes(components / "native_native" / f"{scene}_boxes.pkl"), matrix
    )
    for box in native:
        box2d(ax, box, COLORS["native"], width=0.45, alpha=0.13)

    selected = np.asarray(case["box_aligned"], dtype=float)
    color = COLORS["failure"] if failure else COLORS[case["source"]]
    if show_selected:
        box2d(
            ax,
            selected,
            color,
            width=2.4,
            outlined=case["source"] == "calr" and not failure,
        )
    if case.get("gt_box_aligned") is not None:
        box2d(
            ax,
            case["gt_box_aligned"],
            COLORS["gt"],
            dashed=True,
            width=2.0,
        )

    lo, hi = selected[:, :2].min(0), selected[:, :2].max(0)
    # Tight target-level crop: the full room is context, not evidence.
    margin = max(float((hi - lo).max()) * crop_scale, min_margin)
    center = (lo + hi) / 2
    ax.set_xlim(center[0] - margin, center[0] + margin)
    ax.set_ylim(center[1] - margin, center[1] + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(background_color)
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.55)


def annotation(panel, case, evidence):
    if panel == "mvsr_reranking":
        return (
            f"Native score: {case['original_score']:.3f} \u2192 {case['reranked_score']:.3f}\n"
            f"Rank among native records:\n{case['original_rank']} \u2192 {case['reranked_rank']}\n"
            f"Historical support: {evidence['support']:.3f}"
        )
    if panel == "failure":
        return (
            f"Native score: {case['original_score']:.3f} \u2192 {case['reranked_score']:.3f}\n"
            f"Rank among native records:\n{case['original_rank']} \u2192 {case['reranked_rank']}\n"
            f"Historical support: {evidence['support']:.3f}"
        )
    return f"3D AABB IoU = {case['iou']:.2f}"


def add_metric_box(ax, text, *, failure=False):
    ax.text(
        0.975,
        0.965,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.7,
        linespacing=1.35,
        color="#172033",
        bbox={
            "boxstyle": "round,pad=0.34",
            "facecolor": "white",
            "edgecolor": COLORS["failure"] if failure else "#94A3B8",
            "linewidth": 0.75,
            "alpha": 0.94,
        },
    )


def comparison_panel(fig, cell, panel, case, evidence, args):
    grid = cell.subgridspec(2, 1, height_ratios=(0.78, 1.22), hspace=0.13)
    rgb = fig.add_subplot(grid[0])
    rgb_strip(rgb, case, evidence, args.raw_root)
    rgb.set_title(
        TITLES[panel], fontsize=8.3, fontweight="bold", loc="left", pad=3
    )

    maps = grid[1].subgridspec(1, 2, wspace=0.06)
    base_ax = fig.add_subplot(maps[0])
    recar_ax = fig.add_subplot(maps[1])
    topdown(
        base_ax,
        case,
        args.components,
        args.gt_root,
        args.raw_root,
        show_selected=False,
    )
    topdown(recar_ax, case, args.components, args.gt_root, args.raw_root)
    base_ax.set_title(
        "BoxFusion-Ctrl", fontsize=7.0, fontweight="semibold", pad=2
    )
    recar_ax.set_title("ReCaR-3D", fontsize=7.0, fontweight="semibold", pad=2)
    recar_ax.text(
        0.5,
        0.025,
        annotation(panel, case, evidence),
        transform=recar_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=6.5,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
            "pad": 1.5,
        },
    )


def analysis_panel(fig, cell, panel, case, evidence, args):
    grid = cell.subgridspec(2, 1, height_ratios=(0.78, 1.22), hspace=0.13)
    rgb = fig.add_subplot(grid[0])
    is_failure = panel == "failure"
    rgb_strip(
        rgb,
        case,
        evidence,
        args.raw_root,
        failure=is_failure,
        slots=2,
        card_text=annotation(panel, case, evidence),
        max_frames=1,
    )
    rgb.set_title(
        TITLES[panel], fontsize=8.3, fontweight="bold", loc="left", pad=3
    )
    maps = grid[1].subgridspec(1, 2, wspace=0.06)
    before_ax = fig.add_subplot(maps[0])
    after_ax = fig.add_subplot(maps[1])
    topdown(
        before_ax,
        case,
        args.components,
        args.gt_root,
        args.raw_root,
        failure=is_failure,
        crop_scale=1.72 if is_failure else 1.55,
        min_margin=0.84 if is_failure else 0.72,
        point_darken=0.52 if is_failure else 0.72,
        point_alpha=0.82 if is_failure else 0.52,
        point_size=0.62 if is_failure else 0.28,
        monochrome_points=is_failure,
        background_color="#EEF2F6" if is_failure else "#F8FAFC",
    )
    topdown(
        after_ax,
        case,
        args.components,
        args.gt_root,
        args.raw_root,
        failure=is_failure,
        crop_scale=1.72 if is_failure else 1.55,
        min_margin=0.84 if is_failure else 0.72,
        point_darken=0.52 if is_failure else 0.72,
        point_alpha=0.82 if is_failure else 0.52,
        point_size=0.62 if is_failure else 0.28,
        monochrome_points=is_failure,
        background_color="#EEF2F6" if is_failure else "#F8FAFC",
    )
    before_ax.set_title("Before HCSR", fontsize=7.0, fontweight="semibold", pad=2)
    after_ax.set_title("After HCSR", fontsize=7.0, fontweight="semibold", pad=2)
    if is_failure:
        after_ax.text(
            0.5,
            0.03,
            "No GT match at\nIoU threshold 0.15",
            transform=after_ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.15,
            color=COLORS["failure"],
            fontweight="semibold",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.88,
                "pad": 1.5,
            },
        )


def save_case_panels(payload, evidence, panels, args):
    for panel in panels:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(7.2, 3.2),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.45, 1]},
        )
        rgb_strip(
            axes[0],
            payload["cases"][panel],
            evidence[panel],
            args.raw_root,
            panel == "failure",
        )
        topdown(
            axes[1],
            payload["cases"][panel],
            args.components,
            args.gt_root,
            args.raw_root,
            panel == "failure",
            crop_scale=1.72 if panel == "failure" else 1.25,
            min_margin=0.84 if panel == "failure" else 0.55,
            point_alpha=0.82 if panel == "failure" else 0.52,
            point_size=0.62 if panel == "failure" else 0.28,
            monochrome_points=panel == "failure",
            background_color=(
                "#EEF2F6" if panel == "failure" else "#F8FAFC"
            ),
        )
        title = TITLES[panel].splitlines()[0]
        scene = payload["cases"][panel]["scene"]
        fig.suptitle(f"{title} \u2014 {scene}", fontsize=11, fontweight="bold")
        axes[1].set_title(
            annotation(panel, payload["cases"][panel], evidence[panel]),
            fontsize=8,
        )
        fig.savefig(
            args.output / "cases" / f"{panel}_2x2_source.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--mvsr-no-visibility",
        type=Path,
        default=DEFAULT_NO_VISIBILITY_COMPONENTS,
        help="Final three-factor MVSR component directory (visibility removed).",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.selection.read_text())
    panels = (
        "plr_recovery",
        "calr_recovery",
        "mvsr_reranking",
        "failure",
    )
    evidence = {}
    for panel in panels:
        case = payload["cases"][panel]
        path = Path(case["evidence_export"])
        if not case.get("evidence_verified") or digest(path) != case["evidence_sha256"]:
            raise RuntimeError(f"unverified qualitative evidence: {panel}")
        evidence[panel] = json.loads(path.read_text())

    # The locked RGB/process traces predate the final no-visibility MVSR
    # ablation. Preserve those verified frames, but bind the displayed score,
    # rank, and support values to the final fixed-geometry component files.
    payload = copy.deepcopy(payload)
    evidence = copy.deepcopy(evidence)
    metrics = {}
    # Bind both sides of the score/rank comparison to the same final ablation
    # run; older locked qualitative traces can differ at float round-off level.
    native_root = args.mvsr_no_visibility.parent / "native_native"
    for panel in ("mvsr_reranking", "failure"):
        row = final_mvsr_metrics(
            payload["cases"][panel], native_root, args.mvsr_no_visibility
        )
        metrics[panel] = row
        payload["cases"][panel].update(
            {
                "original_score": row["original_score"],
                "reranked_score": row["reranked_score"],
                "original_rank": row["original_rank"],
                "reranked_rank": row["reranked_rank"],
            }
        )
        evidence[panel]["support"] = row["max_three_factor_support"]

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases").mkdir(exist_ok=True)
    (args.output / "mvsr_metrics_no_visibility.json").write_text(
        json.dumps(
            {
                "schema": "boxfusion.qualitative.mvsr_no_visibility.v1",
                "mvsr_factors": [
                    "proposal_score",
                    "2d_overlap",
                    "3d_consistency",
                ],
                "temporal_aggregation": "historical_maximum",
                "rank_scope": "within-scene native boxes only",
                "score_transform": "logit(p')=logit(p)+2*max(0,S_max-0.5)",
                "cases": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    save_case_panels(payload, evidence, panels, args)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    ):
        fig = plt.figure(figsize=(183 / 25.4, 5.55), constrained_layout=False)
        outer = fig.add_gridspec(
            2,
            2,
            left=0.035,
            right=0.985,
            bottom=0.03,
            top=0.89,
            wspace=0.11,
            hspace=0.20,
        )
        comparison_panel(
            fig,
            outer[0, 0],
            "plr_recovery",
            payload["cases"]["plr_recovery"],
            evidence["plr_recovery"],
            args,
        )
        comparison_panel(
            fig,
            outer[0, 1],
            "calr_recovery",
            payload["cases"]["calr_recovery"],
            evidence["calr_recovery"],
            args,
        )
        analysis_panel(
            fig,
            outer[1, 0],
            "mvsr_reranking",
            payload["cases"]["mvsr_reranking"],
            evidence["mvsr_reranking"],
            args,
        )
        analysis_panel(
            fig,
            outer[1, 1],
            "failure",
            payload["cases"]["failure"],
            evidence["failure"],
            args,
        )

        legend_handles = [
            Line2D([0], [0], color=COLORS["native"], lw=2.2, label="Native box"),
            Line2D(
                [0], [0], color=COLORS["plr"], lw=2.2,
                label="PLR evidence/recovery"
            ),
            Line2D(
                [0], [0], color=COLORS["calr"], lw=2.2,
                label="CALR evidence/recovery"
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["gt"],
                lw=2.0,
                linestyle="--",
                label="Ground truth",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["support"],
                lw=1.8,
                linestyle="--",
                label="Matched proposal",
            ),
            Line2D(
                [0], [0], color=COLORS["failure"], lw=2.2, label="False positive"
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.982),
            ncol=6,
            frameon=False,
            fontsize=6.15,
            handlelength=1.9,
            columnspacing=0.72,
        )

        base = args.output / "qualitative_comparison_2x2"
        fig.savefig(base.with_suffix(".png"), dpi=600)
        fig.savefig(base.with_suffix(".pdf"))
        fig.savefig(base.with_suffix(".svg"))
        fig.savefig(
            base.with_suffix(".tiff"),
            dpi=600,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        plt.close(fig)

    zh = (
        "图4｜ReCaR-3D的定性结果与代表性失败案例。a，PLR利用多个关键帧中空间一致的post-NMS proposals恢复BoxFusion-Ctrl遗漏的目标。b，CALR通过pre-NMS anchors的跨关键帧体素共识恢复弱目标。c，HCSR根据历史最大三因素复合支持更新原生记录的分数和相对排名；前后视图使用同一原生记录及相同三维几何。d，HCSR的代表性失败案例：偶然获得较高复合支持的误检被错误提升；该框在3D AABB IoU阈值0.15下无匹配GT。蓝色、橙色和深金色分别表示原生框、PLR证据/恢复框和CALR证据/恢复框，绿色虚线表示真实标注，紫色虚线表示匹配的二维辅助proposal，红色表示失败案例中的误检框。最终HCSR复合支持由proposal分数、二维重叠和三维一致性组成，不包含visibility。HCSR本身不生成或删除原生记录，也不修改其几何、身份或语义标签。场景点云仅用于可视化，不作为模型输入。"
    )
    en = (
        "Figure 4. Qualitative results and a representative failure of ReCaR-3D. "
        "(a) PLR recovers a native-map miss from spatially consistent post-NMS proposals "
        "across multiple keyframes. (b) CALR recovers a weak target through cross-keyframe "
        "voxel consensus of pre-NMS anchors. (c) HCSR updates a native record's score and "
        "relative rank using the historical maximum three-factor composite support; the before/after "
        "views contain the same native record with identical 3D geometry. (d) A representative HCSR "
        "failure: a false positive that accidentally receives high composite support is "
        "incorrectly promoted at the 3D AABB IoU threshold of 0.15. Blue, orange, and dark gold "
        "denote native, PLR evidence/recovery, and CALR evidence/recovery, respectively; dashed green denotes ground truth, dashed purple "
        "denotes a matched auxiliary proposal in 2D, and red denotes the false positive in the "
        "failure case. The final HCSR support combines proposal score, 2D overlap, and 3D "
        "consistency; visibility is excluded. HCSR itself does not create or delete native records "
        "and does not alter their geometry, identity, or semantic labels. Scene "
        "point clouds are used only for visualization and are not model inputs."
    )
    (args.output / "caption_zh_2x2.txt").write_text(zh + "\n")
    (args.output / "caption_en_2x2.txt").write_text(en + "\n")


if __name__ == "__main__":
    main()
