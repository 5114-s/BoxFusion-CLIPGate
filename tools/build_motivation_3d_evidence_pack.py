#!/usr/bin/env python3
"""Build real 3D assets for the temporal-evidence motivation figure.

The scene points and cuboids are loaded from frozen ScanNet/evidence outputs.
Only camera projection, cropping, line colour, and figure composition are added.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path("/data/ZhaoX/BoxFusion")
RAW = Path("/extra/ZhaoX/scannet_data/scans")
DATA = ROOT / "evaluation/data_util/scannet_train_detection_data"
REPORT = ROOT / "development/final_pipeline/reports/qualitative_recar3d_final"
OUT = ROOT / "figure_assets/motivation_temporal_evidence_3d_20261101"

BLUE = "#2878C7"
BLUE_LIGHT = "#79AEE3"
PURPLE = "#8552B8"
ORANGE = "#F06B16"
AMBER = "#D99A00"
GREEN = "#1A9A61"
GREY = "#64748B"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def axis_alignment(scene: str) -> np.ndarray:
    path = RAW / scene / f"{scene}.txt"
    for line in path.read_text().splitlines():
        if line.startswith("axisAlignment"):
            return np.asarray(line.split("=", 1)[1].split(), dtype=float).reshape(4, 4)
    raise RuntimeError(f"No axisAlignment in {path}")


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def cuboid_edges(box: np.ndarray) -> list[tuple[int, int]]:
    points = np.asarray(box, dtype=float).reshape(8, 3)
    centered = points - points.mean(axis=0)
    opposite_cost = np.linalg.norm(centered + centered[0], axis=1)
    opposite_cost[0] = np.inf
    opposite = int(np.argmin(opposite_cost))
    others = [index for index in range(8) if index not in (0, opposite)]
    diagonal = points[opposite] - points[0]
    best, best_cost = None, np.inf
    for neighbours in itertools.combinations(others, 3):
        basis = np.stack([points[index] - points[0] for index in neighbours])
        if abs(float(np.linalg.det(basis))) < 1e-12:
            continue
        generated = np.asarray([
            points[0] + np.asarray(code, dtype=float) @ basis
            for code in itertools.product((0, 1), repeat=3)
        ])
        nearest = np.argmin(np.linalg.norm(generated[:, None] - points[None], axis=2), axis=1)
        if len(set(nearest.tolist())) != 8:
            continue
        reconstruction = float(np.max(np.linalg.norm(generated - points[nearest], axis=1)))
        diagonal_error = float(np.linalg.norm(basis.sum(axis=0) - diagonal))
        cost = reconstruction + diagonal_error
        if cost < best_cost:
            best_cost = cost
            best = dict(zip(itertools.product((0, 1), repeat=3), nearest.tolist()))
    if best is None or best_cost > 1e-4:
        raise ValueError(f"cannot recover cuboid corner topology; error={best_cost}")
    return sorted({
        tuple(sorted((best[first], best[second])))
        for first in best for second in best
        if sum(a != b for a, b in zip(first, second)) == 1
    })


def load_cloud(scene: str, centre: np.ndarray, radius_xy: float = 1.75, z_pad: float = .35):
    path = DATA / f"{scene}_vert.npy"
    arr = np.load(path)
    xyz = arr[:, :3].astype(float)
    rgb = np.clip(arr[:, 3:6].astype(float) / 255.0, 0, 1)
    mask = (
        (np.abs(xyz[:, 0] - centre[0]) <= radius_xy)
        & (np.abs(xyz[:, 1] - centre[1]) <= radius_xy)
        & (xyz[:, 2] >= max(-.15, centre[2] - 1.45 - z_pad))
        & (xyz[:, 2] <= min(2.8, centre[2] + 1.45 + z_pad))
    )
    xyz, rgb = xyz[mask], rgb[mask]
    if len(xyz) > 11500:
        ids = np.linspace(0, len(xyz) - 1, 11500, dtype=int)
        xyz, rgb = xyz[ids], rgb[ids]
    return xyz, rgb, path


def draw_box(ax, box: np.ndarray, colour: str, *, lw: float = 2.6, alpha: float = 1.0,
             linestyle: str = "-", zorder: int = 10):
    for i, j in cuboid_edges(box):
        ax.plot(
            [box[i, 0], box[j, 0]], [box[i, 1], box[j, 1]], [box[i, 2], box[j, 2]],
            color=colour, lw=lw, alpha=alpha, linestyle=linestyle, zorder=zorder,
        )


def setup_axis(ax, xyz: np.ndarray, centre: np.ndarray, *, rgb: np.ndarray | None = None,
               radius_xy: float = 1.65,
               elev: float = 25, azim: float = -57):
    colours = "#A8B0BA" if rgb is None else np.clip(rgb * .72 + .28, 0, 1)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=.60, c=colours, alpha=.38,
               depthshade=False, rasterized=True)
    ax.set_xlim(centre[0] - radius_xy, centre[0] + radius_xy)
    ax.set_ylim(centre[1] - radius_xy, centre[1] + radius_xy)
    ax.set_zlim(max(-.10, centre[2] - 1.25), min(2.65, centre[2] + 1.45))
    ax.set_box_aspect((2 * radius_xy, 2 * radius_xy, ax.get_zlim()[1] - ax.get_zlim()[0]))
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.grid(False)
    ax.set_facecolor((1, 1, 1, 0))


def save_single(scene: str, box_specs: list[dict], out: Path, centre: np.ndarray, *,
                title: str = "", subtitle: str = "", radius_xy: float = 1.65,
                centres: list[tuple[np.ndarray, str]] | None = None,
                voxel: np.ndarray | None = None, elev: float = 25, azim: float = -57):
    xyz, rgb, _path = load_cloud(scene, centre, radius_xy + .15)
    fig = plt.figure(figsize=(6.2, 5.0), dpi=260)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=.02, top=.87)
    setup_axis(ax, xyz, centre, rgb=rgb, radius_xy=radius_xy, elev=elev, azim=azim)
    for spec in box_specs:
        draw_box(ax, spec["box"], spec["colour"], lw=spec.get("lw", 2.6),
                 alpha=spec.get("alpha", 1.0), linestyle=spec.get("linestyle", "-"),
                 zorder=spec.get("zorder", 10))
    if voxel is not None:
        draw_box(ax, voxel, "#3A7DE0", lw=2.2, alpha=.95, linestyle="--", zorder=15)
    if centres:
        for point, colour in centres:
            ax.scatter([point[0]], [point[1]], [point[2]], s=72, c=[colour],
                       edgecolors="white", linewidths=1.3, depthshade=False, zorder=30)
    if title:
        fig.suptitle(title, y=.985, fontsize=18, fontweight="bold", color="#17243A")
    if subtitle:
        fig.text(.5, .905, subtitle, ha="center", fontsize=10.5, color=GREY)
    fig.savefig(out, dpi=300, transparent=True)
    plt.close(fig)


def make_hcsr(selected: dict) -> list[Path]:
    case = selected["mvsr_reranking"]
    scene = case["scene"]
    matrix = axis_alignment(scene)
    native = transform(np.asarray(case["box"]), matrix)
    evidence = json.loads(Path(case["evidence_export"]).read_text())
    proposal = transform(np.asarray(evidence["frames"][0]["candidate_box"]), matrix)
    centre = native.mean(0)
    final_values = json.loads((ROOT / "figure_assets/hcsr_qualitative_pair_20261031/source_manifest.json").read_text())["cases"]["mvsr_reranking"]
    before = OUT / "10a_hcsr_3d_before_scene0644_00.png"
    after = OUT / "10b_hcsr_3d_after_scene0644_00.png"
    common = [
        {"box": proposal, "colour": PURPLE, "lw": 2.4, "linestyle": "--", "alpha": .95},
    ]
    save_single(scene, common + [{"box": native, "colour": BLUE_LIGHT, "lw": 2.5}], before, centre,
                title="Before HCSR", subtitle=f"Native score {final_values['original_score']:.3f}  |  rank {final_values['original_rank']}",
                radius_xy=1.22, elev=23, azim=-63)
    save_single(scene, common + [{"box": native, "colour": BLUE, "lw": 3.2}], after, centre,
                title="After HCSR", subtitle=f"Native score {final_values['reranked_score']:.3f}  |  rank {final_values['reranked_rank']}  |  identical geometry",
                radius_xy=1.22, elev=23, azim=-63)

    fig = plt.figure(figsize=(12.2, 5.3), dpi=250)
    for idx, (title, score, rank, colour, lw) in enumerate([
        ("Before HCSR", final_values["original_score"], final_values["original_rank"], BLUE_LIGHT, 2.5),
        ("After HCSR", final_values["reranked_score"], final_values["reranked_rank"], BLUE, 3.2),
    ], 1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        xyz, rgb, _ = load_cloud(scene, centre, 1.37)
        setup_axis(ax, xyz, centre, rgb=rgb, radius_xy=1.22, elev=23, azim=-63)
        draw_box(ax, proposal, PURPLE, lw=2.4, linestyle="--")
        draw_box(ax, native, colour, lw=lw)
        ax.set_title(f"{title}\nscore {score:.3f} · rank {rank}", fontsize=14, fontweight="bold", color="#17243A", pad=2)
    fig.subplots_adjust(left=.01, right=.99, bottom=.12, top=.90, wspace=.00)
    handles = [Line2D([0], [0], color=BLUE, lw=3, label="Native 3D box (same geometry)"),
               Line2D([0], [0], color=PURPLE, lw=2.5, ls="--", label="Matched proposal 3D box")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11)
    out = OUT / "10_hcsr_3d_support_and_reranking_scene0644_00.png"
    fig.savefig(out, dpi=300, transparent=True)
    plt.close(fig)
    return [out, before, after]


def raw_voxel_corners(key: list[int], delta: float) -> np.ndarray:
    lo = np.asarray(key, dtype=float) * delta
    hi = lo + delta
    return np.asarray([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def make_calr(selected: dict) -> list[Path]:
    case = selected["calr_recovery"]
    scene = case["scene"]
    matrix = axis_alignment(scene)
    evidence = json.loads(Path(case["evidence_export"]).read_text())
    rows = evidence["frames"]
    candidates = [transform(np.asarray(row["candidate_box"]), matrix) for row in rows]
    final_box = transform(np.asarray(case["box"]), matrix)
    centre = final_box.mean(0)
    colours = ["#E74B16", "#2AAF69", "#347DDB"]
    singles = []
    for idx, (row, box, colour) in enumerate(zip(rows, candidates, colours), 1):
        out = OUT / f"20{chr(96 + idx)}_calr_3d_anchor_t{idx}_frame{row['frame_id']:04d}.png"
        save_single(scene, [{"box": box, "colour": AMBER, "lw": 2.8}], out, centre,
                    title=f"t{idx}  |  frame {row['frame_id']}", subtitle="Measured lifted pre-NMS anchor",
                    radius_xy=1.18, centres=[(box.mean(0), colour)], elev=26, azim=-52)
        singles.append(out)

    key = evidence["terminal_trace"]["voxel_key"]
    voxel = transform(raw_voxel_corners(key, .30), matrix)
    consensus = OUT / "20d_calr_3d_world_voxel_consensus_scene0606_01.png"
    centre_dots = [(box.mean(0), colour) for box, colour in zip(candidates, colours)]
    save_single(scene, [{"box": final_box, "colour": AMBER, "lw": 3.0}], consensus, centre,
                title="World-space voxel consensus", subtitle="Three distinct keyframes vote for one 0.30 m voxel",
                radius_xy=1.18, centres=centre_dots, voxel=voxel, elev=26, azim=-52)

    fig = plt.figure(figsize=(15.0, 4.2), dpi=240)
    xyz, rgb, _ = load_cloud(scene, centre, 1.33)
    specs = [(f"t{i+1}\nframe {row['frame_id']}", box, colour) for i, (row, box, colour) in enumerate(zip(rows, candidates, colours))]
    for idx, (title, box, colour) in enumerate(specs, 1):
        ax = fig.add_subplot(1, 4, idx, projection="3d")
        setup_axis(ax, xyz, centre, rgb=rgb, radius_xy=1.18, elev=26, azim=-52)
        draw_box(ax, box, AMBER, lw=2.7)
        c = box.mean(0)
        ax.scatter([c[0]], [c[1]], [c[2]], s=60, c=[colour], edgecolors="white", linewidths=1.1, depthshade=False)
        ax.set_title(title, fontsize=12, fontweight="bold", color="#17243A", pad=0)
    ax = fig.add_subplot(1, 4, 4, projection="3d")
    setup_axis(ax, xyz, centre, rgb=rgb, radius_xy=1.18, elev=26, azim=-52)
    draw_box(ax, voxel, "#3A7DE0", lw=2.2, linestyle="--")
    draw_box(ax, final_box, AMBER, lw=3.0)
    for point, colour in centre_dots:
        ax.scatter([point[0]], [point[1]], [point[2]], s=60, c=[colour], edgecolors="white", linewidths=1.1, depthshade=False)
    ax.set_title("Voxel consensus\nrepresentative anchor", fontsize=12, fontweight="bold", color="#17243A", pad=0)
    fig.subplots_adjust(left=.005, right=.995, bottom=.07, top=.87, wspace=-.03)
    fig.text(.5, .025, "Actual lifted anchors in one world-coordinate scene; the dashed cube is the measured 0.30 m voting cell.",
             ha="center", fontsize=10, color=GREY)
    combined = OUT / "20_calr_3d_temporal_consensus_scene0606_01.png"
    fig.savefig(combined, dpi=300, transparent=True)
    plt.close(fig)
    return [combined, *singles, consensus]


def copy_plr_assets() -> list[Path]:
    source = ROOT / "figure_assets/real_3d_motivation_20261026"
    mapping = {
        "01_native_object_map_3d_scene0426_03.png": ROOT / "figure_assets/real_3d_motivation_optimized_20261028/native_scene_3d_color.png",
        "02_plr_3d_temporal_evidence_scene0426_03.png": source / "three_evidence_scenes_3d.png",
        "02a_plr_3d_t1_frame0425.png": source / "evidence_3d_frame_000425.png",
        "02b_plr_3d_t2_frame0450.png": source / "evidence_3d_frame_000450.png",
        "02c_plr_3d_t3_frame0550.png": source / "evidence_3d_frame_000550.png",
        "03_plr_3d_native_vs_recovered_scene0426_03.png": ROOT / "figure_assets/real_3d_motivation_optimized_20261028/optimized_3d_comparison.png",
        "03b_plr_3d_native_vs_recovered_vector.svg": ROOT / "figure_assets/real_3d_motivation_optimized_20261028/optimized_3d_comparison.svg",
    }
    outputs = []
    for name, src in mapping.items():
        dst = OUT / name
        shutil.copy2(src, dst)
        outputs.append(dst)
    return outputs


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGBA")
    fitted = ImageOps.contain(im, size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", size, (248, 250, 253, 255))
    canvas.alpha_composite(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def make_index(items: list[tuple[str, Path]]) -> Path:
    card_w, card_h = 740, 520
    cols = 2
    rows = (len(items) + 1) // 2
    margin, gap, head = 34, 26, 92
    canvas = Image.new("RGBA", (margin * 2 + cols * card_w + gap, head + margin + rows * (card_h + gap)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((margin, 18), "3D temporal-evidence assets (real frozen outputs)", fill="#17243A", font=font(40, True))
    for idx, (title, path) in enumerate(items):
        row, col = divmod(idx, 2)
        x, y = margin + col * (card_w + gap), head + row * (card_h + gap)
        d.rounded_rectangle((x, y, x + card_w, y + card_h), radius=16, fill="#F8FAFD", outline="#C4CFDD", width=3)
        d.text((x + 18, y + 12), title, fill="#17243A", font=font(27, True))
        canvas.alpha_composite(fit_image(path, (card_w - 30, card_h - 65)), (x + 15, y + 52))
    out = OUT / "00_3d_asset_index.png"
    canvas.convert("RGB").save(out, dpi=(220, 220))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected_path = REPORT / "selected_cases.json"
    selected = json.loads(selected_path.read_text())["cases"]
    plr = copy_plr_assets()
    hcsr = make_hcsr(selected)
    calr = make_calr(selected)
    index = make_index([
        ("Native object-map state", OUT / "01_native_object_map_3d_scene0426_03.png"),
        ("PLR: proposal evidence across time", OUT / "02_plr_3d_temporal_evidence_scene0426_03.png"),
        ("PLR: map-only vs recovered", OUT / "03_plr_3d_native_vs_recovered_scene0426_03.png"),
        ("HCSR: same 3D geometry, new rank", OUT / "10_hcsr_3d_support_and_reranking_scene0644_00.png"),
        ("CALR: lifted anchors and voxel state", OUT / "20_calr_3d_temporal_consensus_scene0606_01.png"),
        ("CALR: consensus detail", OUT / "20d_calr_3d_world_voxel_consensus_scene0606_01.png"),
    ])

    readme = """# 3D assets for the ReCaR-3D motivation figure

All scene backgrounds are real ScanNet geometry. All displayed cuboids are measured frozen outputs. No model was rerun and no generative image was used.

## Recommended use

- Upper baseline row: `01_native_object_map_3d...png` represents the accepted-object map state. Use the three individual PLR evidence images without temporal links to indicate that auxiliary observations are available frame by frame but are not retained as a cross-keyframe state by the controlled baseline.
- Upper ReCaR-3D row: use `02_plr_3d_temporal_evidence...png` and add a vector association line across t1/t2/t3. Add a small CALR voxel-state glyph cropped from `20d...png`.
- HCSR case: use `10_hcsr_3d_support_and_reranking...png`. The native cuboid is geometrically identical before/after; only the native score/rank changes. Purple dashed is the matched proposal cuboid.
- PLR case: use `03_plr_3d_native_vs_recovered...png`, or assemble the individual `02a/02b/02c` images.
- CALR case: use `20_calr_3d_temporal_consensus...png`; the first three panels are actual lifted pre-NMS anchors and the final dashed blue cube is the actual 0.30 m world-space voting voxel derived from key `[8,16,2]`.

Do not imply that the static ScanNet geometry is an input to ReCaR-3D. It is only a common visualization background. HCSR, PLR and CALR examples are selected independently from different scenes.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    sources = {
        "selected_cases": {"path": str(selected_path), "sha256": sha256(selected_path)},
        "hcsr_evidence": {"path": selected["mvsr_reranking"]["evidence_export"], "sha256": sha256(Path(selected["mvsr_reranking"]["evidence_export"]))},
        "calr_evidence": {"path": selected["calr_recovery"]["evidence_export"], "sha256": sha256(Path(selected["calr_recovery"]["evidence_export"]))},
        "plr_3d_manifest": {"path": str(ROOT / "figure_assets/real_3d_motivation_20261026/source_manifest.json"), "sha256": sha256(ROOT / "figure_assets/real_3d_motivation_20261026/source_manifest.json")},
    }
    manifest = {
        "schema": "recar3d.motivation_3d_evidence.v1",
        "created": "2026-11-01",
        "generative_images_used": False,
        "new_model_inference": False,
        "background_geometry_is_visualization_only": True,
        "scenes": {"plr": "scene0426_03", "hcsr": "scene0644_00", "calr": "scene0606_01"},
        "sources": sources,
        "outputs": {},
    }
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.name not in {"source_manifest.json", "source_hashes.sha256"}]
    manifest["outputs"] = {p.name: sha256(p) for p in sorted(outputs)}
    (OUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    checks = [f"{sha256(p)}  {p.name}" for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "source_hashes.sha256"]
    (OUT / "source_hashes.sha256").write_text("\n".join(checks) + "\n")
    print(index)
    print(f"Built {len(list(OUT.iterdir()))} files in {OUT}")


if __name__ == "__main__":
    main()
