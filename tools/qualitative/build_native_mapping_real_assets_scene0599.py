#!/usr/bin/env python3
"""Export audited real assets for the Native Mapping Path panel.

The panel uses one coherent online prefix from ScanNet scene0599_01:

* previous native state: keyframe 225;
* current per-frame candidates and Boxer lifting: keyframe 250;
* current native state: keyframe 250.

The RGB-D prefix point cloud is visualization only.  Every blue cuboid is an
unchanged native-record or Boxer-output OBB from the archived experiment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SCENE = "scene0599_01"
PREVIOUS_FRAME = 225
CURRENT_FRAME = 250
REPRESENTATIVE_CANDIDATE = 0
OUT = ROOT / "figure_assets/native_mapping_path_verified_real_scene0599_kf250_20261107"
ZIP = OUT.with_suffix(".zip")
RGB_ROOT = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color"
DEPTH_ROOT = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/depth"
NATIVE_CACHE = (
    ROOT
    / "development/final_controls/experiments/mvsr_multiview_audit_20261012/native_prefix"
    / f"{SCENE}.npz"
)
BOXER_LOG = (
    ROOT
    / "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009"
    / "receipts/recar"
    / SCENE
    / "native_boxer_diagnostics"
    / f"{SCENE}_boxer_lifting.jsonl"
)
SCAN_ROOT = Path("/extra/ZhaoX/scannet_data/scans") / SCENE
READOUT_MANIFEST = (
    ROOT
    / "figure_assets/online_detection_readout_real_dual_branch_20261107/source_manifest.json"
)
BLUE = "#236FB6"
INK = "#142033"
CORNER_CODES = np.asarray(list(itertools.product((-.5, .5), repeat=3)), dtype=float)
EDGE_INDEX = np.asarray([
    (i, j)
    for i in range(8)
    for j in range(i + 1, 8)
    if bin(i ^ j).count("1") == 1
], dtype=int)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def axis_alignment() -> np.ndarray:
    line = next(
        row for row in (SCAN_ROOT / f"{SCENE}.txt").read_text().splitlines()
        if row.startswith("axisAlignment")
    )
    return np.asarray(line.split("=", 1)[1].split(), dtype=float).reshape(4, 4)


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def native_snapshots() -> tuple[dict[int, dict], np.ndarray]:
    with np.load(NATIVE_CACHE, allow_pickle=False) as values:
        arrays = {key: np.asarray(values[key]) for key in values.files}
    offsets = np.r_[0, np.cumsum(arrays["native_lengths"])]
    snapshots: dict[int, dict] = {}
    for index, frame in enumerate(arrays["frame_ids"]):
        lo, hi = int(offsets[index]), int(offsets[index + 1])
        snapshots[int(frame)] = {
            "ids": arrays["native_ids"][lo:hi],
            "corners": arrays["native_corners"][lo:hi],
            "scores": arrays["native_scores"][lo:hi],
            "pose": arrays["camera_to_world"][index],
            "intrinsic": arrays["intrinsic"][index],
            "width": int(arrays["width"][index]),
            "height": int(arrays["height"][index]),
        }
    return snapshots, arrays["frame_ids"]


def boxer_row(frame: int) -> dict:
    for line in BOXER_LOG.open():
        row = json.loads(line)
        if int(row["frame_id"]) == frame:
            return row
    raise KeyError(frame)


def boxer_corners_camera(row: dict) -> np.ndarray:
    boxes = np.asarray(row["output_xyz_dims_camera"], dtype=float)
    rotations = np.asarray(row["output_rotation_camera_object"], dtype=float)
    local = CORNER_CODES[None] * boxes[:, None, 3:]
    # rotation_camera_object stores object axes as columns.  For row vectors,
    # local-to-camera is local @ R.T.
    return boxes[:, None, :3] + np.einsum("nkj,nij->nki", local, rotations)


def project_camera(corners: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = corners[..., 2]
    uv = np.empty(corners.shape[:-1] + (2,), dtype=float)
    uv[..., 0] = intrinsic[0, 0] * corners[..., 0] / z + intrinsic[0, 2]
    uv[..., 1] = intrinsic[1, 1] * corners[..., 1] / z + intrinsic[1, 2]
    return uv


def draw_rectangles(image: Image.Image, boxes: np.ndarray) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    halo, width = 11, 6
    for box in boxes:
        coords = tuple(float(x) for x in box)
        draw.rectangle(coords, outline="white", width=halo)
        draw.rectangle(coords, outline=BLUE, width=width)
    return out


def draw_wireframes(image: Image.Image, corners_uv: np.ndarray) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for corners in corners_uv:
        for i, j in EDGE_INDEX:
            segment = [tuple(corners[i]), tuple(corners[j])]
            draw.line(segment, fill="white", width=10)
            draw.line(segment, fill=BLUE, width=6)
    return out


def crop_around(image: Image.Image, points: np.ndarray, pad_ratio: float = .10) -> Image.Image:
    pts = np.asarray(points).reshape(-1, 2)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    pts[:, 0] = np.clip(pts[:, 0], 0, image.width)
    pts[:, 1] = np.clip(pts[:, 1], 0, image.height)
    x1, y1 = pts.min(0)
    x2, y2 = pts.max(0)
    pad = max(x2 - x1, y2 - y1) * pad_ratio
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(image.width, x2 + pad), min(image.height, y2 + pad)
    # Keep a readable landscape crop for the compact method panel.
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width, height = x2 - x1, y2 - y1
    target = 1.48
    if width / max(height, 1) < target:
        width = height * target
    else:
        height = width / target
    return image.crop((
        round(max(0, cx - width / 2)), round(max(0, cy - height / 2)),
        round(min(image.width, cx + width / 2)), round(min(image.height, cy + height / 2)),
    ))


def prefix_point_cloud(
    end_frame: int,
    frame_ids: np.ndarray,
    snapshots: dict[int, dict],
    alignment: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points, colors = [], []
    for frame in frame_ids[frame_ids <= end_frame]:
        frame = int(frame)
        state = snapshots[frame]
        depth = np.asarray(Image.open(DEPTH_ROOT / f"{frame}.png"), dtype=float) / 1000.0
        rgb = Image.open(RGB_ROOT / f"{frame}.jpg").convert("RGB").resize(
            (state["width"], state["height"]), Image.Resampling.BILINEAR
        )
        rgb = np.asarray(rgb, dtype=float) / 255.0
        step = 3
        vv, uu = np.mgrid[0:state["height"]:step, 0:state["width"]:step]
        zz = depth[::step, ::step]
        valid = (zz > .20) & (zz < 5.0)
        u, v, z = uu[valid], vv[valid], zz[valid]
        intrinsic = state["intrinsic"]
        camera = np.column_stack((
            (u - intrinsic[0, 2]) * z / intrinsic[0, 0],
            (v - intrinsic[1, 2]) * z / intrinsic[1, 1],
            z,
        ))
        pose = state["pose"]
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        points.append(transform(world, alignment))
        colors.append(rgb[::step, ::step][valid])
    return np.concatenate(points), np.concatenate(colors)


def render_maps(
    previous_points: np.ndarray,
    previous_colors: np.ndarray,
    current_points: np.ndarray,
    current_colors: np.ndarray,
    previous_boxes: np.ndarray,
    current_boxes: np.ndarray,
    renderer,
    *,
    suffix: str = "",
) -> dict[str, Path]:
    camera = json.loads(READOUT_MANIFEST.read_text())["camera"]
    args = (
        np.asarray(camera["position"]), np.asarray(camera["look_at"]),
        float(camera["focal_pixels"]), np.asarray(camera["principal_point_pixels"]),
    )
    projected_sets = []
    for pts, boxes in ((previous_points, previous_boxes), (current_points, current_boxes)):
        pp = renderer.project(pts, *args)
        pb = renderer.project(boxes.reshape(-1, 3), *args)
        projected_sets.append((pp, pb))
    all_uv = np.concatenate([
        projected_sets[0][0][:, :2], projected_sets[0][1][:, :2],
        projected_sets[1][0][:, :2], projected_sets[1][1][:, :2],
    ])
    finite = np.all(np.isfinite(all_uv), axis=1)
    low = np.quantile(all_uv[finite], .002, axis=0) - 25
    high = np.quantile(all_uv[finite], .998, axis=0) + 25
    outputs = {}
    for name, pts, colors, boxes, (projected, _) in (
        (f"previous_native_map_frame_{PREVIOUS_FRAME:04d}{suffix}", previous_points, previous_colors, previous_boxes, projected_sets[0]),
        (f"current_native_map_frame_{CURRENT_FRAME:04d}{suffix}", current_points, current_colors, current_boxes, projected_sets[1]),
    ):
        valid = projected[:, 2] > 0
        order = np.argsort(projected[valid, 2])[::-1]
        uv = projected[valid][order]
        rgb = np.clip(colors[valid][order] ** .88, 0, 1)
        fig = plt.figure(figsize=(7.5, 6), dpi=240)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.scatter(uv[:, 0], uv[:, 1], c=rgb, s=.85, marker=".", linewidths=0,
                   alpha=.92, rasterized=True, zorder=1)
        for box in boxes:
            projected_box = renderer.project(box, *args)
            segments = projected_box[:, :2][renderer.box_edges(box)]
            halo = LineCollection(segments, colors="white", linewidths=4.1,
                                  alpha=.92, zorder=8)
            lines = LineCollection(segments, colors=BLUE, linewidths=2.25,
                                   alpha=.99, zorder=9)
            ax.add_collection(halo); ax.add_collection(lines)
        ax.set_xlim(low[0], high[0]); ax.set_ylim(high[1], low[1])
        ax.set_aspect("equal"); ax.axis("off")
        white = OUT / f"{name}_white.png"
        transparent = OUT / f"{name}_transparent.png"
        fig.savefig(white, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=.01)
        fig.savefig(transparent, dpi=400, transparent=True, facecolor="none",
                    bbox_inches="tight", pad_inches=.01)
        plt.close(fig)
        outputs[name] = white
    return outputs


def render_mesh_maps(
    previous_boxes: np.ndarray,
    current_boxes: np.ndarray,
    renderer,
    alignment: np.ndarray,
) -> dict[str, Path]:
    """Render the exact native records over the real ScanNet mesh."""
    camera = json.loads(READOUT_MANIFEST.read_text())["camera"]
    xyz, _, display_rgb, faces, _ = renderer.load_mesh(
        SCENE, alignment, [14, 15, 16, 18, 20]
    )
    args = (
        np.asarray(camera["position"]), np.asarray(camera["look_at"]),
        float(camera["focal_pixels"]), np.asarray(camera["principal_point_pixels"]),
    )
    projected = renderer.project(xyz, *args)
    background = renderer.rasterize(
        projected, display_rgb, faces, *camera["image_size"]
    )
    limits = camera["display_limits"]
    outputs = {}
    for name, boxes in (
        (f"previous_native_records_mesh_frame_{PREVIOUS_FRAME:04d}", previous_boxes),
        (f"current_native_records_mesh_frame_{CURRENT_FRAME:04d}", current_boxes),
    ):
        fig = plt.figure(figsize=(7.5, 6), dpi=240)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(background, interpolation="antialiased")
        for box in boxes:
            projected_box = renderer.project(box, *args)
            segments = projected_box[:, :2][renderer.box_edges(box)]
            ax.add_collection(LineCollection(
                segments, colors="white", linewidths=4.1, alpha=.91, zorder=8
            ))
            ax.add_collection(LineCollection(
                segments, colors=BLUE, linewidths=2.25, alpha=.99, zorder=9
            ))
        ax.set_xlim(limits[:2]); ax.set_ylim(limits[2:])
        ax.set_aspect("equal"); ax.axis("off")
        white = OUT / f"{name}_white.png"
        transparent = OUT / f"{name}_transparent.png"
        fig.savefig(white, dpi=400, facecolor="white", bbox_inches="tight", pad_inches=.01)
        fig.savefig(transparent, dpi=400, transparent=True, facecolor="none",
                    bbox_inches="tight", pad_inches=.01)
        plt.close(fig)
        outputs[name] = white
    return outputs


def contact_sheet(items: list[tuple[str, Path]], path: Path) -> None:
    width, image_h, label_h = 760, 570, 54
    canvas = Image.new("RGB", (width * 2, (image_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, source) in enumerate(items):
        image = Image.open(source).convert("RGBA")
        fitted = ImageOps.contain(image, (width - 28, image_h - 24), Image.Resampling.LANCZOS)
        x = index % 2 * width + (width - fitted.width) // 2
        y = index // 2 * (image_h + label_h) + (image_h - fitted.height) // 2
        canvas.paste(fitted, (x, y), fitted)
        draw.text((index % 2 * width + 18, index // 2 * (image_h + label_h) + image_h + 7),
                  label, fill=INK, font=font)
    canvas.save(path)


def box_iou_2d(left: np.ndarray, right: np.ndarray) -> float:
    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    a = max(0., left[2] - left[0]) * max(0., left[3] - left[1])
    b = max(0., right[2] - right[0]) * max(0., right[3] - right[1])
    return inter / max(a + b - inter, 1e-12)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots, frame_ids = native_snapshots()
    if PREVIOUS_FRAME not in snapshots or CURRENT_FRAME not in snapshots:
        raise RuntimeError("Required native prefix snapshots are missing")
    row = boxer_row(CURRENT_FRAME)
    if int(row["count"]) != len(row["input_boxes_xyxy"]):
        raise RuntimeError("Boxer diagnostic count mismatch")
    current = snapshots[CURRENT_FRAME]
    image = Image.open(RGB_ROOT / f"{CURRENT_FRAME}.jpg").convert("RGB")
    scale = np.asarray([image.width / current["width"], image.height / current["height"]] * 2)
    input_boxes = np.asarray(row["input_boxes_xyxy"], dtype=float) * scale
    corners_camera = boxer_corners_camera(row)
    corners_uv_small = project_camera(corners_camera, current["intrinsic"])
    corners_uv = corners_uv_small * np.asarray([image.width / current["width"], image.height / current["height"]])

    # Keep all-candidate panels for audit, while the compact framework panel
    # uses one unchanged representative chair candidate for legibility.
    all_two_d = draw_rectangles(image, input_boxes)
    all_three_d = draw_wireframes(image, corners_uv)
    all_two_d.save(OUT / f"frame_candidates_2d_frame_{CURRENT_FRAME:04d}_all.png")
    all_three_d.save(OUT / f"boxer_lifted_3d_frame_{CURRENT_FRAME:04d}_all.png")
    selected_box = input_boxes[REPRESENTATIVE_CANDIDATE:REPRESENTATIVE_CANDIDATE + 1]
    selected_corners = corners_uv[REPRESENTATIVE_CANDIDATE:REPRESENTATIVE_CANDIDATE + 1]
    two_d = draw_rectangles(image, selected_box)
    three_d = draw_wireframes(image, selected_corners)
    crop_around(two_d, selected_box.reshape(-1, 2), pad_ratio=.15).save(
        OUT / f"frame_candidate_2d_frame_{CURRENT_FRAME:04d}.png"
    )
    crop_around(three_d, selected_corners, pad_ratio=.15).save(
        OUT / f"boxer_lifted_3d_frame_{CURRENT_FRAME:04d}.png"
    )
    image.save(OUT / f"rgb_frame_{CURRENT_FRAME:04d}.png")
    Image.open(DEPTH_ROOT / f"{CURRENT_FRAME}.png").save(
        OUT / f"depth_frame_{CURRENT_FRAME:04d}_raw.png"
    )

    alignment = axis_alignment()
    previous_points, previous_colors = prefix_point_cloud(
        PREVIOUS_FRAME, frame_ids, snapshots, alignment
    )
    current_points, current_colors = prefix_point_cloud(
        CURRENT_FRAME, frame_ids, snapshots, alignment
    )
    previous_boxes_all = transform(snapshots[PREVIOUS_FRAME]["corners"], alignment)
    current_boxes_all = transform(snapshots[CURRENT_FRAME]["corners"], alignment)
    # A compact, identity-consistent readout for the small framework slot.
    # These five high-score identities persist across the two states; ID 41 is
    # the new record at the current keyframe.  The complete states are exported
    # separately below for audit.
    persistent_display_ids = [0, 20, 1, 21, 12]
    current_display_ids = persistent_display_ids + [41]
    previous_id_to_row = {
        int(identity): row
        for row, identity in enumerate(snapshots[PREVIOUS_FRAME]["ids"])
    }
    current_id_to_row = {
        int(identity): row
        for row, identity in enumerate(snapshots[CURRENT_FRAME]["ids"])
    }
    if not all(identity in previous_id_to_row for identity in persistent_display_ids):
        raise RuntimeError("A selected persistent identity is missing from the previous state")
    if not all(identity in current_id_to_row for identity in current_display_ids):
        raise RuntimeError("A selected display identity is missing from the current state")
    previous_boxes = previous_boxes_all[
        [previous_id_to_row[identity] for identity in persistent_display_ids]
    ]
    current_boxes = current_boxes_all[
        [current_id_to_row[identity] for identity in current_display_ids]
    ]
    renderer = load_module(
        "native_mapping_real_renderer",
        ROOT / "tools/qualitative/render_real_3d_motivation.py",
    )
    maps = render_maps(
        previous_points, previous_colors, current_points, current_colors,
        previous_boxes, current_boxes, renderer,
    )
    render_maps(
        previous_points, previous_colors, current_points, current_colors,
        previous_boxes_all, current_boxes_all, renderer,
        suffix="_all_records",
    )
    mesh_maps = render_mesh_maps(
        previous_boxes, current_boxes, renderer, alignment,
    )

    # Audit that the projected Boxer cuboids are consistent with their exact
    # 2D proposal rows.  No matching is performed: the archived row order is
    # preserved by the lifting contract.
    projected_bounds = np.column_stack((
        corners_uv_small[..., 0].min(1), corners_uv_small[..., 1].min(1),
        corners_uv_small[..., 0].max(1), corners_uv_small[..., 1].max(1),
    ))
    clipped = projected_bounds.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, current["width"])
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, current["height"])
    ious = [
        box_iou_2d(np.asarray(box), projected)
        for box, projected in zip(row["input_boxes_xyxy"], clipped)
    ]

    contact_sheet([
        (f"Exact 2D frame candidate · keyframe {CURRENT_FRAME}", OUT / f"frame_candidate_2d_frame_{CURRENT_FRAME:04d}.png"),
        (f"Exact Boxer 3D OBB projection · keyframe {CURRENT_FRAME}", OUT / f"boxer_lifted_3d_frame_{CURRENT_FRAME:04d}.png"),
        (f"Previous native records · keyframe {PREVIOUS_FRAME}", mesh_maps[f"previous_native_records_mesh_frame_{PREVIOUS_FRAME:04d}"]),
        (f"Current native records · keyframe {CURRENT_FRAME}", mesh_maps[f"current_native_records_mesh_frame_{CURRENT_FRAME:04d}"]),
    ], OUT / "00_contact_sheet.png")

    manifest = {
        "schema": "carr3d.native_mapping_path.real_assets.v2",
        "scene": SCENE,
        "previous_keyframe": PREVIOUS_FRAME,
        "current_keyframe": CURRENT_FRAME,
        "native_record_counts": {
            "previous": int(len(previous_boxes_all)), "current": int(len(current_boxes_all)),
        },
        "framework_display": {
            "previous_native_identity_ids": persistent_display_ids,
            "current_native_identity_ids": current_display_ids,
            "selection": "identity-consistent high-score native records; coordinates unchanged",
        },
        "current_frame_candidate_count": int(len(input_boxes)),
        "representative_candidate_row": REPRESENTATIVE_CANDIDATE,
        "boxer_3d_obb_count": int(len(corners_camera)),
        "boxer_projection_iou_with_input_2d": ious,
        "boxer_projection_median_iou": float(np.median(ious)),
        "sources": {
            "native_prefix_cache": str(NATIVE_CACHE),
            "native_prefix_sha256": sha256(NATIVE_CACHE),
            "boxer_diagnostic": str(BOXER_LOG),
            "boxer_diagnostic_sha256": sha256(BOXER_LOG),
            "rgb": str(RGB_ROOT / f"{CURRENT_FRAME}.jpg"),
            "depth": str(DEPTH_ROOT / f"{CURRENT_FRAME}.png"),
            "axis_alignment": str(SCAN_ROOT / f"{SCENE}.txt"),
        },
        "geometry_edited": False,
        "generative_images_used": False,
        "point_cloud_note": "RGB-D prefix points are visualization only; the method remains reconstruction-free.",
        "mesh_note": "The complete ScanNet scene mesh is visualization only and is not maintained by the reconstruction-free method.",
        "correction_to_old_pack": "The old 'native_boxer_3d_projection' panel displayed resized Boxer-input 2D rectangles. This pack projects the archived output_xyz_dims_camera and output_rotation_camera_object OBBs.",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "README_CN.md").write_text(
        f"# Native Mapping Path 真实图片包\n\n"
        f"同一在线前缀：`{SCENE}`，上一关键帧 {PREVIOUS_FRAME}，当前关键帧 {CURRENT_FRAME}。\n\n"
        "- `frame_candidate_2d_*`：流程图使用的真实二维候选示例。\n"
        "- `boxer_lifted_3d_*`：同一候选的真实 Boxer OBB 透视投影。\n"
        "- 文件名含 `all` 的图片保留该关键帧的全部候选，用于审计。\n"
        f"- `previous_native_map_*`：关键帧 {PREVIOUS_FRAME} 的 5 条持续高分原生记录。\n"
        f"- `current_native_map_*`：同一组记录加当前新增身份，共 6 条。\n"
        f"- 文件名含 `all_records` 的图片展示完整 {len(previous_boxes_all)}/{len(current_boxes_all)} 条内部记录。\n\n"
        "- `*_records_mesh_*`：论文流程图推荐版本，使用真实 ScanNet 网格作为可视化背景。\n\n"
        "地图中的 RGB-D 前缀点云和场景网格仅用于可视化，不是方法维护的重建状态。"
        "所有蓝框均来自归档实验，未修改坐标、尺寸或朝向。\n",
        encoding="utf-8",
    )

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(OUT.iterdir()):
            if path.is_file():
                archive.write(path, Path(OUT.name) / path.name)
    print(json.dumps({
        "folder": str(OUT), "zip": str(ZIP),
        "previous_records": len(previous_boxes_all), "current_records": len(current_boxes_all),
        "candidate_count": len(input_boxes), "median_projection_iou": float(np.median(ious)),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
