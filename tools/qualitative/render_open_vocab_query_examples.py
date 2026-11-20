#!/usr/bin/env python3
"""Render measured frozen-CLIP text readout examples for ReCaR-3D.

The figure intentionally uses 3D AABBs rather than fabricated instance masks.
Every highlighted record comes from the archived final ReCaR-3D prediction
file, and every query label/cosine comes from the frozen 60-query CLIP cache.
GT is used only to audit and select correctly localized illustrative records.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.qualitative import render_boxfusion_ctrl_recar_3d as render

OUT = ROOT / "figure_assets/open_vocab_query_readout_20261104"
SEMANTIC = ROOT / "reports/scannet_zero_shot_open_vocab_boxfusion_original_20261010"

SCENES = (
    {
        "scene": "scene0606_01", "gt_indices": [0, 2, 3, 4, 5], "walls": [1, 2],
        "records": (
            {"index": 15, "query": "table", "source": "Native"},
            {"index": 32, "query": "picture", "source": "PLR"},
            {"index": 67, "query": "chair", "source": "CALR"},
        ),
    },
    {
        "scene": "scene0300_00", "gt_indices": [2, 3, 5, 8], "walls": [14, 15],
        "records": (
            {"index": 2, "query": "chair", "source": "Native"},
            {"index": 7, "query": "monitor", "source": "PLR"},
            {"index": 128, "query": "table", "source": "CALR"},
        ),
    },
)

SOURCE_COLORS = {"Native": "#2878B5", "PLR": "#F07820", "CALR": "#B27D00"}
RED = "#D8262F"
INK = "#14283F"
MUTED = "#607086"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def muted_mesh(image: np.ndarray) -> np.ndarray:
    source = Image.fromarray(image)
    if source.mode == "RGBA":
        white = Image.new("RGBA", source.size, (255, 255, 255, 255))
        source = Image.alpha_composite(white, source).convert("RGB")
    else:
        source = source.convert("RGB")
    gray = ImageOps.grayscale(source).convert("RGB")
    gray = ImageEnhance.Contrast(gray).enhance(0.86)
    white = Image.new("RGB", gray.size, (255, 255, 255))
    return np.asarray(Image.blend(gray, white, 0.12))


def audit_records(payload: dict, spec: dict, vocabulary: list[str]) -> list[dict]:
    scene = spec["scene"]
    pred = payload["predictions"]["ReCaR-3D"]
    native_n = len(payload["predictions"]["BoxFusion-Ctrl"]["boxes"])
    with np.load(SEMANTIC / "recar/semantic_cache" / f"{scene}.npz") as semantic:
        labels = semantic["labels"].copy()
        cosine = semantic["max_cosine"].copy()
        view_ids = semantic["view_ids"].copy()
        rectangles = semantic["rectangles"].copy()
    with np.load(SEMANTIC / "gt_cache" / f"{scene}.npz") as gt:
        gt_boxes = gt["boxes"].copy()
        gt_classes = gt["classes"].copy()
    overlaps = render.iou_matrix(pred["boxes"], gt_boxes)
    records = []
    for requested in spec["records"]:
        index = requested["index"]
        gt_index = int(np.argmax(overlaps[index]))
        branch = pred["branches"][index]
        expected_branch = {"Native": "native", "PLR": "plr", "CALR": "calr"}[requested["source"]]
        label = int(labels[index])
        assert branch == expected_branch
        assert vocabulary[label] == requested["query"]
        assert label == int(gt_classes[gt_index])
        record_iou = float(overlaps[index, gt_index])
        native_best = float(overlaps[:native_n, gt_index].max())
        assert record_iou >= 0.25
        if requested["source"] != "Native":
            assert native_best < 0.15
        records.append({
            **requested,
            "branch": branch,
            "gt_index": gt_index,
            "iou": record_iou,
            "native_best_iou": native_best,
            "cosine": float(cosine[index]),
            "view_id": int(view_ids[index]),
            "rectangle": rectangles[index].tolist(),
            "detection_score": float(pred["scores"][index]),
        })
    return records


def panel(ax, payload: dict, renderer, *, gray: bool) -> None:
    ax.imshow(muted_mesh(payload["background"]) if gray else payload["background"])
    ax.set_xlim(payload["camera"]["display_limits"][:2])
    ax.set_ylim(payload["camera"]["display_limits"][2:])
    ax.axis("off")


def export_figure(payloads: list[dict], audits: list[list[dict]], renderer) -> list[str]:
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.45))
    fig.subplots_adjust(left=.038, right=.992, bottom=.145, top=.915, wspace=.025, hspace=.31)
    headers = ("Scene mesh", "Native-record query", "PLR-recovery query", "CALR-recovery query")
    for column, title in enumerate(headers):
        bounds = axes[0, column].get_position()
        fig.text(bounds.x0 + bounds.width / 2, .955, title, ha="center", va="center",
                 fontsize=12.2, fontweight="bold", color=INK)
    for row, (payload, records) in enumerate(zip(payloads, audits)):
        scene = payload["spec"]["scene"]
        for column, ax in enumerate(axes[row]):
            panel(ax, payload, renderer, gray=column > 0)
            if column == 0:
                ax.text(.03, .95, f"({chr(97 + row)}) {scene}", transform=ax.transAxes,
                        ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK,
                        bbox={"facecolor": "white", "alpha": .83, "edgecolor": "none", "pad": 2.2})
                ax.text(.5, -.06, "Reconstructed scene mesh", transform=ax.transAxes,
                        ha="center", va="top", fontsize=10.2, color=INK)
                continue
            item = records[column - 1]
            box = payload["predictions"]["ReCaR-3D"]["boxes"][item["index"]]
            render.draw_box(ax, box, renderer, payload["projection_args"], RED, lw=2.35, alpha=1.0)
            ax.text(.035, .94, item["source"], transform=ax.transAxes, ha="left", va="top",
                    fontsize=9.1, fontweight="bold", color="white",
                    bbox={"boxstyle": "round,pad=.24", "facecolor": SOURCE_COLORS[item["source"]],
                          "edgecolor": "none", "alpha": .96})
            ax.text(.5, -.05, f'“{item["query"]}”', transform=ax.transAxes,
                    ha="center", va="top", fontsize=12.0, color=INK, fontweight="bold")
            qualifier = (f'cos {item["cosine"]:.3f} · 3D IoU {item["iou"]:.2f}'
                         if item["source"] == "Native" else
                         f'cos {item["cosine"]:.3f} · IoU {item["iou"]:.2f} · native miss')
            ax.text(.5, -.125, qualifier, transform=ax.transAxes,
                    ha="center", va="top", fontsize=7.7, color=MUTED)
    legend = [
        Line2D([], [], color=RED, lw=2.4, label="Selected terminal 3D record"),
        *[Line2D([], [], color=color, lw=5, label=f"{source} source")
          for source, color in SOURCE_COLORS.items()],
    ]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .055), ncol=4,
               frameon=False, fontsize=8.2, handlelength=2.0, columnspacing=1.7)
    fig.text(.5, .018,
             "Frozen CLIP argmax over the shared 60-query vocabulary; red denotes a 3D AABB, not an instance mask.",
             ha="center", color=MUTED, fontsize=8.0)
    files = []
    for suffix in ("png", "pdf", "svg"):
        name = f"open_vocab_terminal_query_examples.{suffix}"
        fig.savefig(OUT / name, dpi=400 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
        files.append(name)
    plt.close(fig)
    return files


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "svg.fonttype": "none", "text.color": INK})
    vocabulary = json.loads((SEMANTIC / "split.json").read_text())["all_names"]
    renderer = render.load_renderer()
    payloads, audits = [], []
    for spec in SCENES:
        payload = render.scene_payload(spec, renderer, OUT, raster_scale=1)
        records = audit_records(payload, spec, vocabulary)
        payloads.append(payload)
        audits.append(records)
        print(f"audited {spec['scene']}: " + ", ".join(r["query"] for r in records), flush=True)
    files = export_figure(payloads, audits, renderer)
    manifest = {
        "schema": "recar3d.open_vocab_terminal_query_examples.v1",
        "protocol": "frozen CLIP ViT-H-14; argmax normalized cosine over shared 60-query vocabulary",
        "selection": "GT-assisted selection of correctly localized illustrative records; predictions and semantic labels unchanged",
        "claim_boundary": "qualitative compatibility/readout example, not a base/novel generalization experiment",
        "instance_masks_used": False,
        "highlight": "measured 3D AABB",
        "scenes": [{"scene": spec["scene"], "records": records,
                    "prediction_path": payload["predictions"]["ReCaR-3D"]["path"],
                    "prediction_sha256": payload["predictions"]["ReCaR-3D"]["sha256"],
                    "camera": payload["camera"]}
                   for spec, payload, records in zip(SCENES, payloads, audits)],
        "semantic_split_sha256": digest(SEMANTIC / "split.json"),
        "script_sha256": digest(Path(__file__)),
        "artifacts": {name: digest(OUT / name) for name in files},
    }
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    caption = (
        "图X｜冻结开放词汇语义读出的三维查询示例。每行第一列为重建场景网格，后三列分别展示来自原生记录、PLR恢复记录与CALR恢复记录的终态文本读出。"
        "查询词为冻结CLIP在统一60查询词表上的真实argmax输出，红色为对应三维AABB，而非实例分割掩码。"
        "scene0606_01中的picture与chair，以及scene0300_00中的monitor与table，在原生输出中与相应GT的最大三维IoU均低于0.15；候选恢复后得到图示记录。"
        "所示记录经GT辅助选择以验证几何正确性，但预测框、查询标签与余弦相似度均未修改。该图说明恢复记录能够继承冻结语义接口，不构成base/novel泛化实验。\n"
    )
    (OUT / "中文图注.txt").write_text(caption)
    print(json.dumps({"output": str(OUT), "files": files}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
