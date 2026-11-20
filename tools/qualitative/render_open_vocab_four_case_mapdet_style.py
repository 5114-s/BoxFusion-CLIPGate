#!/usr/bin/env python3
"""Render four audited CaRR-3D open-vocabulary cases in a Map-Det3D layout.

The reference layout groups a 2x2 RGB grid on the left and a matching 2x2
3D-output grid on the right.  This script reuses the verified semantic-readout
loader and renderer; it does not rerun inference or alter prediction geometry.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
BASE_SCRIPT = ROOT / "tools/qualitative/render_open_vocab_readout_carr_clean.py"
OUT = ROOT / "figure_assets/open_vocab_four_case_mapdet_style_tnr_20261113"

INK = "#101820"
RED = "#D62828"
RULE = "#CBD2DA"

# The first two cases come from the existing two-case main figure.  The final
# two are audited recovery records already used by the frozen-CLIP query figure.
CASES = (
    {
        "row": "a",
        "branch": "PTR",
        "branch_key": "ptr",
        "source_key": "m1p",
        "scene": "scene0426_03",
        "prediction_index": 25,
        "frame_id": 575,
        "query": "window",
        "view_crop": [216, 58, 1296, 868],
    },
    {
        "row": "b",
        "branch": "VAR",
        "branch_key": "var",
        "source_key": "m1a",
        "scene": "scene0591_01",
        "prediction_index": 55,
        "frame_id": 1250,
        "query": "trash can",
        "display_query": "trash",
        "view_crop": [0, 356, 760, 926],
    },
    {
        "row": "c",
        "branch": "PTR",
        "branch_key": "ptr",
        "source_key": "m1p",
        "scene": "scene0164_02",
        "prediction_index": 15,
        "frame_id": 275,
        "query": "microwave",
        "view_crop": [142, 75, 902, 645],
    },
    {
        "row": "d",
        "branch": "CALR",
        "branch_key": "calr",
        "source_key": "m1a",
        "scene": "scene0693_00",
        "prediction_index": 23,
        "frame_id": 1175,
        "query": "bottle",
        "view_crop": [46, 335, 846, 935],
    },
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def load_base_module():
    spec = importlib.util.spec_from_file_location("carr_readout_base", BASE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def frame(ax) -> None:
    ax.add_patch(
        Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            fill=False,
            edgecolor=RULE,
            linewidth=0.55,
            zorder=100,
            clip_on=False,
        )
    )


def draw_rgb(ax, payload: dict) -> None:
    ax.imshow(payload["rgb"], interpolation="antialiased")
    x1, y1, x2, y2 = payload["rect"]
    outline = Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False,
        edgecolor=RED,
        linewidth=1.15,
        zorder=8,
        path_effects=[
            pe.Stroke(linewidth=2.0, foreground="white", alpha=0.78),
            pe.Normal(),
        ],
    )
    ax.add_patch(outline)
    ax.set_xlim(0, payload["rgb"].shape[1])
    ax.set_ylim(payload["rgb"].shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    frame(ax)
    ax.text(
        0.025, 0.955,
        f"({payload['row']})  “{payload.get('display_query', payload['query'])}”",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 1.5},
        zorder=20,
    )


def draw_rgb_3d_output(ax, payload: dict, base) -> None:
    """Draw the 3D cuboid over the exact same RGB pixels as its input pair."""
    ax.imshow(payload["rgb"], interpolation="antialiased")
    base.draw_projected_box(
        ax,
        payload,
        payload["recovery"],
        RED,
        width=1.45,
        zorder=12,
    )
    ax.set_xlim(0, payload["rgb"].shape[1])
    ax.set_ylim(payload["rgb"].shape[0], 0)
    ax.set_aspect("equal")
    ax.axis("off")
    frame(ax)


def render(base, payloads: list[dict]) -> list[Path]:
    # 183 mm is a typical two-column paper width.  The approximately 2:1
    # aspect ratio follows the supplied Map-Det3D reference figure.
    width_inches = 183 / 25.4
    fig = plt.figure(figsize=(width_inches, 3.72), facecolor="white")
    grid = fig.add_gridspec(
        2, 5,
        left=0.018,
        right=0.992,
        bottom=0.135,
        top=0.985,
        width_ratios=[1.0, 1.0, 0.12, 1.0, 1.0],
        wspace=0.030,
        hspace=0.085,
    )

    # Same spatial order in both groups makes the RGB-to-3D correspondence
    # immediate: (a,b) on the first row and (c,d) on the second row.
    for index, payload in enumerate(payloads):
        row, local_col = divmod(index, 2)
        rgb_ax = fig.add_subplot(grid[row, local_col])
        draw_rgb(rgb_ax, payload)

        output_ax = fig.add_subplot(grid[row, local_col + 3])
        draw_rgb_3d_output(output_ax, payload, base)

    left_a = grid[0, 0].get_position(fig)
    left_b = grid[0, 1].get_position(fig)
    right_a = grid[0, 3].get_position(fig)
    right_b = grid[0, 4].get_position(fig)
    fig.text(
        (left_a.x0 + left_b.x1) / 2,
        0.055,
        "RGB Input",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        (right_a.x0 + right_b.x1) / 2,
        0.055,
        "3D Output",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="semibold",
        color=INK,
    )

    outputs = []
    for suffix in ("png", "pdf", "svg"):
        path = OUT / f"carr3d_open_vocab_four_case_mapdet_style.{suffix}"
        fig.savefig(path, dpi=500 if suffix == "png" else None, facecolor="white")
        outputs.append(path)
    preview = OUT / "carr3d_open_vocab_four_case_mapdet_style_preview.png"
    fig.savefig(preview, dpi=180, facecolor="white")
    plt.close(fig)
    return [*outputs, preview]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            # Times New Roman is preferred.  Liberation Serif is the metrically
            # compatible fallback on this Linux host; the editable SVG retains
            # the full family list so Windows/Word resolves Times New Roman.
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman",
                "Liberation Serif",
                "serif",
            ],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.55,
            "text.color": INK,
        }
    )

    base = load_base_module()
    renderer = base.load_renderer()
    payloads = []
    for case in CASES:
        payload = base.load_case(case, renderer)
        payloads.append(payload)
        print(
            f"audited ({case['row']}) {case['scene']} / {case['query']}: "
            f"top1={payload['top1']} IoU={payload['iou']:.4f}",
            flush=True,
        )

    outputs = render(base, payloads)
    font_path = Path(findfont(FontProperties(family="serif")))
    source_hashes = {}
    for payload in payloads:
        for key in (
            "full_path",
            "native_path",
            "diagnostic_path",
            "semantic_path",
            "gt_path",
            "rgb_path",
            "pose_path",
            "intrinsic_path",
        ):
            source = Path(payload[key])
            source_hashes[str(source)] = digest(source)

    manifest = {
        "schema": "carr3d.open_vocab_four_case.mapdet_style.v1",
        "layout_reference": "Map-Det3D qualitative figure: 2x2 RGB group plus 2x2 3D-output group",
        "semantic_protocol": (
            "Frozen CLIP ViT-H-14, fixed 60-query vocabulary, maximum valid "
            "projected view; semantic readout applied to archived terminal boxes"
        ),
        "cases": [
            {
                "panel": payload["row"],
                "scene": payload["scene"],
                "query": payload["query"],
                "display_query": payload.get("display_query", payload["query"]),
                "top1": payload["top1"],
                "branch": payload["branch"],
                "source_key": payload["source_key"],
                "prediction_index": payload["prediction_index"],
                "frame_id": payload["frame_id"],
                "cosine": payload["cosine"],
                "recovery_iou": payload["iou"],
                "native_best_iou": payload["native_best_iou"],
                "semantic_rectangle_projection_max_error_pixels": payload["projection_error"],
                "rgb_rectangle_xyxy": payload["rect"].tolist(),
                "output_view_crop_xyxy": payload["view_crop"],
            }
            for payload in payloads
        ],
        "typography": {
            "requested": "Times New Roman",
            "family_stack": plt.rcParams["font.serif"],
            "host_raster_font": str(font_path),
            "editable_svg_text": True,
        },
        "paired_panel_invariant": {
            "rgb_background_pixels": "identical between each input/output pair",
            "camera": "identical selected RGB camera and color intrinsics",
            "input_overlay": "projected 2D query rectangle",
            "output_overlay": "projected terminal 3D cuboid",
            "color_processing": "none",
        },
        "source_sha256": source_hashes,
        "render_script": str(Path(__file__)),
        "render_script_sha256": digest(Path(__file__)),
        "outputs": {path.name: digest(path) for path in outputs},
        "model_inference_rerun": False,
        "prediction_coordinates_edited": False,
        "generative_images_used": False,
        "claim_boundary": "selected audited examples; not unseen-category generalization evidence",
    }
    manifest_path = OUT / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    caption_path = OUT / "caption_en.txt"
    caption_path.write_text(
        "Qualitative open-vocabulary readout on ScanNetV2. Given a text query "
        "and an RGB observation, CaRR-3D retrieves the corresponding terminal "
        "3D record. Each input/output pair uses the same RGB frame and color; "
        "red denotes the selected 2D query region or projected 3D bounding box.\n",
        encoding="utf-8",
    )
    readme_path = OUT / "README.txt"
    readme_path.write_text(
        "The PNG, PDF, and editable SVG contain four audited CaRR-3D open-vocabulary cases.\n"
        "Panel order is identical in the RGB and 3D groups: window, trash, microwave, bottle.\n"
        "Each paired panel uses the exact same RGB pixels and camera; only the overlay changes from a 2D rectangle to a 3D cuboid.\n"
        "The displayed label 'trash' is an alias; the archived frozen-vocabulary query remains 'trash can'.\n"
        "No model inference was rerun and no prediction coordinates were modified.\n"
        "The SVG keeps live serif text with Times New Roman first in the family stack.\n",
        encoding="utf-8",
    )

    bundle = OUT.with_suffix(".zip")
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, f"{OUT.name}/{path.relative_to(OUT)}")

    print(
        json.dumps(
            {
                "figure": str(outputs[0]),
                "preview": str(outputs[-1]),
                "manifest": str(manifest_path),
                "bundle": str(bundle),
                "host_raster_font": str(font_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
