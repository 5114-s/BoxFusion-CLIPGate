#!/usr/bin/env python3
"""Export audited real assets for the HSR, PTR, and VAR framework panels.

The recommended PTR/VAR frame panels show the *lifted 3D boxes actually used by
the online state*, projected back into their source RGB observations.  Separate
2D audit panels retain the detector rectangles.  No box geometry is edited and
no generative image is used.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import pickle
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SCENE = "scene0599_01"
OUT = ROOT / "figure_assets/cross_keyframe_modules_verified_real_scene0599_20261107"
ZIP = OUT.with_suffix(".zip")
RGB_ROOT = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color"
NATIVE = ROOT / "development/final_controls/experiments/mvsr_multiview_audit_20261012/native_prefix" / f"{SCENE}.npz"
EVIDENCE = ROOT / "development/final_controls/experiments/mvsr_unified_samepass_20261010/evidence" / f"{SCENE}.npz"
RAW_ROOT = ROOT / "reports/scannet_seedless_full100_20260911/raw" / SCENE
READOUT = ROOT / "reports/scannet_hcsr_current_zero_vs_max_20261029/frame_readouts" / f"{SCENE}.npz"
RUN = ROOT / "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009"
FULL_PKL = RUN / "receipts/recar" / SCENE / "recar" / f"{SCENE}_boxes.pkl"
OLD_REAL = ROOT / "figure_assets/carr3d_framework_scene0599_real_20261107"
SOURCE = ROOT / "development/mvsr_multiview_audit_20261012_source"

BLUE = "#2374C6"
ORANGE = "#F06418"
GOLD = "#E0A000"
INK = "#142033"

sys.path.insert(0, str(SOURCE))
from boxfusion.native_reliability_reranker import _project_with_visibility  # noqa: E402
from boxfusion.online_candidate_map import _aabb_iou, _iou2d_matrix  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def slices(lengths: np.ndarray) -> list[slice]:
    offsets = np.r_[0, np.cumsum(lengths)]
    return [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(len(lengths))]


def cuboid_edges(box: np.ndarray) -> np.ndarray:
    points = np.asarray(box, dtype=float).reshape(8, 3)
    centered = points - points.mean(axis=0)
    opposite_cost = np.linalg.norm(centered + centered[0], axis=1)
    opposite_cost[0] = np.inf
    opposite = int(np.argmin(opposite_cost))
    others = [i for i in range(8) if i not in (0, opposite)]
    diagonal = points[opposite] - points[0]
    best, best_cost = None, np.inf
    for neighbours in itertools.combinations(others, 3):
        basis = np.stack([points[i] - points[0] for i in neighbours])
        if abs(float(np.linalg.det(basis))) < 1e-12:
            continue
        codes = list(itertools.product((0, 1), repeat=3))
        generated = np.asarray([points[0] + np.asarray(code) @ basis for code in codes])
        nearest = np.argmin(np.linalg.norm(generated[:, None] - points[None], axis=2), axis=1)
        if len(set(nearest.tolist())) != 8:
            continue
        reconstruction = float(np.max(np.linalg.norm(generated - points[nearest], axis=1)))
        diagonal_error = float(np.linalg.norm(basis.sum(axis=0) - diagonal))
        cost = reconstruction + diagonal_error
        if cost < best_cost:
            best_cost = cost
            best = dict(zip(codes, nearest.tolist()))
    if best is None or best_cost > 1e-4:
        raise ValueError(f"cannot recover cuboid topology: {best_cost}")
    edges = {
        tuple(sorted((best[a], best[b])))
        for a in best for b in best
        if sum(x != y for x, y in zip(a, b)) == 1
    }
    if len(edges) != 12:
        raise AssertionError(len(edges))
    return np.asarray(sorted(edges), dtype=int)


def project_world(corners: np.ndarray, pose: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(pose)
    homogeneous = np.concatenate([corners, np.ones(corners.shape[:-1] + (1,))], axis=-1)
    camera = homogeneous @ inverse.T
    z = camera[..., 2]
    uv = np.empty(camera.shape[:-1] + (2,), dtype=float)
    uv[..., 0] = intrinsic[0, 0] * camera[..., 0] / z + intrinsic[0, 2]
    uv[..., 1] = intrinsic[1, 1] * camera[..., 1] / z + intrinsic[1, 2]
    return uv, z


def scale_uv(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(uv, dtype=float).copy()
    value[..., 0] *= width / 640.0
    value[..., 1] *= height / 480.0
    return value


def scale_rect(rect: np.ndarray, width: int, height: int) -> np.ndarray:
    value = np.asarray(rect, dtype=float).copy()
    value[[0, 2]] *= width / 640.0
    value[[1, 3]] *= height / 480.0
    return value


def crop_bounds(points: np.ndarray, width: int, height: int, ratio: float = 1.52) -> tuple[int, int, int, int]:
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, width)
    pts[:, 1] = np.clip(pts[:, 1], 0, height)
    low, high = pts.min(0), pts.max(0)
    center = (low + high) / 2
    span = np.maximum(high - low, [100, 100]) * 1.36
    if span[0] / span[1] < ratio:
        span[0] = span[1] * ratio
    else:
        span[1] = span[0] / ratio
    low = np.maximum(center - span / 2, 0)
    high = np.minimum(center + span / 2, [width, height])
    return tuple(np.r_[low, high].round().astype(int))


def draw_rects(frame: int, rects: list[np.ndarray], colors: list[str], stem: Path, *, native_scale=True) -> tuple[Path, Path]:
    image = Image.open(RGB_ROOT / f"{frame}.jpg").convert("RGB")
    boxes = [scale_rect(r, image.width, image.height) if native_scale else np.asarray(r, float) for r in rects]
    draw = ImageDraw.Draw(image)
    width = max(4, round(image.width / 280))
    for box, color in zip(boxes, colors):
        draw.rectangle(tuple(box), outline="white", width=width + 4)
        draw.rectangle(tuple(box), outline=color, width=width)
    full = stem.with_name(stem.name + "_full.png")
    crop = stem.with_name(stem.name + "_crop.png")
    image.save(full)
    points = np.asarray([[b[0], b[1], b[2], b[3]] for b in boxes]).reshape(-1, 2)
    image.crop(crop_bounds(points, image.width, image.height)).save(crop)
    return full, crop


def draw_wireframe(frame: int, corners: np.ndarray, color: str, stem: Path, pose: np.ndarray, intrinsic: np.ndarray) -> tuple[Path, Path, np.ndarray]:
    image = Image.open(RGB_ROOT / f"{frame}.jpg").convert("RGB")
    uv, z = project_world(np.asarray(corners, float)[None], pose, intrinsic)
    if np.any(z <= 1e-4):
        raise RuntimeError(f"box behind camera at frame {frame}")
    uv = scale_uv(uv[0], image.width, image.height)
    draw = ImageDraw.Draw(image)
    width = max(4, round(image.width / 280))
    for first, second in cuboid_edges(corners):
        segment = [tuple(uv[first]), tuple(uv[second])]
        draw.line(segment, fill="white", width=width + 4)
        draw.line(segment, fill=color, width=width)
    full = stem.with_name(stem.name + "_full.png")
    crop = stem.with_name(stem.name + "_crop.png")
    image.save(full)
    image.crop(crop_bounds(uv, image.width, image.height)).save(crop)
    return full, crop, uv


def exact_assignment(native_corners, proposal_boxes, pose, intrinsic, width, height):
    rows, boxes = [], []
    for row, corners in enumerate(native_corners):
        value = _project_with_visibility(corners, pose, intrinsic, width, height)
        if value is not None:
            rows.append(row); boxes.append(value[0])
    overlap = _iou2d_matrix(np.asarray(boxes), proposal_boxes)
    matches = {}
    pairs = [(float(overlap[r, c]), r, c) for r, c in zip(*np.where(overlap >= .1))]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_r, used_p = set(), set()
    for iou, rr, pr in pairs:
        if rr in used_r or pr in used_p:
            continue
        used_r.add(rr); used_p.add(pr)
        matches[rows[rr]] = (pr, iou, boxes[rr])
    return matches


def read_rows(path: Path):
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload[0] if len(payload) == 1 and isinstance(payload[0], list) else payload


def contact_sheet(items: list[tuple[str, Path]], destination: Path, columns: int = 4):
    tw, th, lh = 430, 285, 44
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (tw * columns, (th + lh) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    try: font = ImageFont.truetype("DejaVuSans.ttf", 21)
    except OSError: font = ImageFont.load_default()
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (tw - 20, th - 20), Image.Resampling.LANCZOS)
        col, row = index % columns, index // columns
        x = col * tw + (tw - image.width) // 2
        y = row * (th + lh) + (th - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((col * tw + 12, row * (th + lh) + th + 6), label, fill=INK, font=font)
    sheet.save(destination)


def draw_hsr_history(frames: list[int], current: list[float], running_max: list[float], destination: Path):
    """Draw a compact plot using the real HSR readout values."""
    width, height = 820, 390
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
        small = ImageFont.truetype("DejaVuSans.ttf", 23)
    except OSError:
        font = small = ImageFont.load_default()
    left, right, top, bottom = 56, width - 42, 30, height - 70
    draw.line((left, bottom, right, bottom), fill="#9A9DA6", width=4)
    xs = np.linspace(left + 80, right - 70, len(frames))
    bar_width = 82
    scale = (bottom - top) / 1.0
    for index, (x, value) in enumerate(zip(xs, current)):
        y = bottom - value * scale
        draw.rounded_rectangle((x - bar_width / 2, y, x + bar_width / 2, bottom), radius=8, fill="#D9C4EE")
        draw.text((x - 22, bottom + 14), f"t{index-len(frames)+1:+d}" if index < len(frames)-1 else "t", fill=INK, font=small)
    points = [(float(xs[0]), bottom - running_max[0] * scale)]
    for index in range(1, len(xs)):
        points.append((float(xs[index]), bottom - running_max[index - 1] * scale))
        points.append((float(xs[index]), bottom - running_max[index] * scale))
    draw.line(points, fill="#7650AA", width=8, joint="curve")
    for x, value in zip(xs, running_max):
        y = bottom - value * scale
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="white", outline="#7650AA", width=6)
    label = f"running max = {running_max[-1]:.3f}"
    draw.text((right - 285, top - 2), label, fill="#7650AA", font=font)
    image.save(destination)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    for folder in ["01_hsr", "02_ptr", "03_var", "04_audit"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    n = np.load(NATIVE, allow_pickle=False)
    e = np.load(EVIDENCE, allow_pickle=False)
    h = np.load(READOUT, allow_pickle=False)
    ns, ps, ass, hs = slices(n["native_lengths"]), slices(e["proposal_lengths"]), slices(e["anchor_lengths"]), slices(h["lengths"])
    frame_index = {int(frame): i for i, frame in enumerate(n["frame_ids"])}
    preview = []

    # HSR: clear table example with an exact exclusive runtime match, a real
    # running-max update, and an actual score/rank change in the HSR readout.
    hsr_frame, hsr_identity = 1800, 321
    fi = frame_index[hsr_frame]
    native_ids = n["native_ids"][ns[fi]]
    native_row = int(np.where(native_ids == hsr_identity)[0][0])
    proposal_boxes = e["proposal_boxes_2d"][ps[fi]]
    matches = exact_assignment(
        n["native_corners"][ns[fi]], proposal_boxes,
        n["camera_to_world"][fi], n["intrinsic"][fi],
        int(n["width"][fi]), int(n["height"][fi]),
    )
    proposal_row, hsr_iou, native_rect = matches[native_row]
    _, hsr_crop = draw_rects(
        hsr_frame, [native_rect, proposal_boxes[proposal_row]], [BLUE, ORANGE],
        OUT / "01_hsr/hsr_record_proposal_match_frame_1800",
    )
    preview.append(("HSR exact match · f1800", hsr_crop))

    hfi = int(np.where(h["frame_ids"] == hsr_frame)[0][0])
    hids = h["ids"][hs[hfi]]
    hrow = int(np.where(hids == hsr_identity)[0][0])
    hj = hs[hfi].start + hrow
    native_scores = h["scores_native"][hs[hfi]]
    reranked_scores = h["scores_historical_max"][hs[hfi]]
    native_rank = int(np.where(np.argsort(-native_scores, kind="stable") == hrow)[0][0]) + 1
    reranked_rank = int(np.where(np.argsort(-reranked_scores, kind="stable") == hrow)[0][0]) + 1
    history_frames = [1725, 1750, 1775, 1800]
    history_current, history_max = [], []
    for frame in history_frames:
        history_fi = int(np.where(h["frame_ids"] == frame)[0][0])
        history_ids = h["ids"][hs[history_fi]]
        history_row = int(np.where(history_ids == hsr_identity)[0][0])
        history_j = hs[history_fi].start + history_row
        history_current.append(float(h["current_support"][history_j]))
        history_max.append(float(h["max_support"][history_j]))
    history_path = OUT / "01_hsr/hsr_identity321_real_history.png"
    draw_hsr_history(history_frames, history_current, history_max, history_path)
    preview.append(("HSR real support history", history_path))

    # PTR: three post-NMS proposals most consistent with unchanged PTR output
    # row 11. Draw the lifted 3D proposal boxes because PTR tracks 3D boxes.
    full_rows = read_rows(FULL_PKL)
    ptr_target = np.asarray(full_rows[11][1], float)
    ptr_entries = []
    ptr_corners = []
    for frame in [0, 25, 50]:
        fi = frame_index[frame]
        corners = e["proposal_corners"][ps[fi]]
        ious = np.asarray([_aabb_iou(ptr_target, box) for box in corners])
        row = int(np.argmax(ious))
        box = corners[row]
        _, crop, _ = draw_wireframe(
            frame, box, ORANGE, OUT / f"02_ptr/ptr_lifted3d_frame_{frame:04d}",
            n["camera_to_world"][fi], n["intrinsic"][fi],
        )
        # 2D detector rectangle, kept for audit only.
        draw_rects(frame, [e["proposal_boxes_2d"][ps[fi]][row]], [ORANGE], OUT / f"04_audit/ptr_detector2d_frame_{frame:04d}")
        ptr_corners.append(box)
        entry = {
            "frame": frame,
            "proposal_row": row,
            "proposal_id": int(e["proposal_ids"][ps[fi]][row]),
            "score": float(e["proposal_scores"][ps[fi]][row]),
            "aabb_iou_to_ptr_output": float(ious[row]),
        }
        ptr_entries.append(entry)
        preview.append((f"PTR lifted proposal · f{frame}", crop))
    for i in range(1, len(ptr_entries)):
        ptr_entries[i]["aabb_iou_to_previous_frame"] = float(_aabb_iou(ptr_corners[i - 1], ptr_corners[i]))
        ptr_entries[i]["center_distance_to_previous_m"] = float(np.linalg.norm(ptr_corners[i - 1].mean(0) - ptr_corners[i].mean(0)))
    shutil.copy2(OLD_REAL / "ptr_recovery.png", OUT / "02_ptr/ptr_recovery_3d.png")
    preview.append(("PTR recovery · output row 11", OUT / "02_ptr/ptr_recovery_3d.png"))

    # VAR: three pre-NMS Top-300 anchors most consistent with unchanged VAR
    # output row 61. Draw lifted 3D anchors. Raw detector rectangles are copied
    # separately without the old erroneous 640x480 rescaling.
    var_target = np.asarray(full_rows[61][1], float)
    var_entries = []
    var_corners = []
    for frame in [825, 850, 875]:
        fi = frame_index[frame]
        corners = e["anchor_corners"][ass[fi]]
        ious = np.asarray([_aabb_iou(var_target, box) for box in corners])
        row = int(np.argmax(ious))
        box = corners[row]
        _, crop, _ = draw_wireframe(
            frame, box, GOLD, OUT / f"03_var/var_lifted3d_frame_{frame:04d}",
            n["camera_to_world"][fi], n["intrinsic"][fi],
        )
        anchor_id = int(e["anchor_ids"][ass[fi]][row])
        with np.load(RAW_ROOT / f"raw_{frame:06d}.npz", allow_pickle=False) as raw:
            raw_rect = np.asarray(raw["boxes"][anchor_id], float)
        draw_rects(frame, [raw_rect], [GOLD], OUT / f"04_audit/var_detector2d_frame_{frame:04d}", native_scale=False)
        var_corners.append(box)
        center = box.mean(0)
        entry = {
            "frame": frame,
            "anchor_row": row,
            "anchor_id": anchor_id,
            "score": float(e["anchor_scores"][ass[fi]][row]),
            "aabb_iou_to_var_output": float(ious[row]),
            "center_x": float(center[0]), "center_y": float(center[1]), "center_z": float(center[2]),
            "voxel_x_0p3m": int(np.floor(center[0] / .3)),
            "voxel_y_0p3m": int(np.floor(center[1] / .3)),
            "voxel_z_0p3m": int(np.floor(center[2] / .3)),
        }
        var_entries.append(entry)
        preview.append((f"VAR lifted anchor · f{frame}", crop))
    for i in range(1, len(var_entries)):
        var_entries[i]["aabb_iou_to_previous_frame"] = float(_aabb_iou(var_corners[i - 1], var_corners[i]))
        var_entries[i]["center_distance_to_previous_m"] = float(np.linalg.norm(var_corners[i - 1].mean(0) - var_corners[i].mean(0)))
    shutil.copy2(OLD_REAL / "var_recovery.png", OUT / "03_var/var_recovery_3d.png")
    preview.append(("VAR recovery · output row 61", OUT / "03_var/var_recovery_3d.png"))

    with (OUT / "04_audit/ptr_track.csv").open("w", newline="") as handle:
        keys = sorted({key for row in ptr_entries for key in row})
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(ptr_entries)
    with (OUT / "04_audit/var_support.csv").open("w", newline="") as handle:
        keys = sorted({key for row in var_entries for key in row})
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(var_entries)

    contact_sheet(preview, OUT / "00_contact_sheet.png")
    manifest = {
        "schema": "carr3d.cross_keyframe_real_assets.v1",
        "scene": SCENE,
        "generative_images_used": False,
        "geometry_edited": False,
        "recommended_figure_assets": {
            "hsr": "01_hsr/hsr_record_proposal_match_frame_1800_crop.png",
            "hsr_history": "01_hsr/hsr_identity321_real_history.png",
            "ptr_frames": [f"02_ptr/ptr_lifted3d_frame_{f:04d}_crop.png" for f in [0, 25, 50]],
            "ptr_recovery": "02_ptr/ptr_recovery_3d.png",
            "var_frames": [f"03_var/var_lifted3d_frame_{f:04d}_crop.png" for f in [825, 850, 875]],
            "var_recovery": "03_var/var_recovery_3d.png",
        },
        "hsr": {
            "frame": hsr_frame, "native_identity": hsr_identity,
            "native_row": native_row, "proposal_row": int(proposal_row),
            "proposal_id": int(e["proposal_ids"][ps[fi]][proposal_row]) if False else int(e["proposal_ids"][ps[frame_index[hsr_frame]]][proposal_row]),
            "iou2d": float(hsr_iou),
            "native_score": float(h["scores_native"][hj]),
            "historical_max_support": float(h["max_support"][hj]),
            "reranked_score": float(h["scores_historical_max"][hj]),
            "native_rank_within_frame": native_rank,
            "reranked_rank_within_frame": reranked_rank,
            "history_frames": history_frames,
            "history_current_support": history_current,
            "history_running_max": history_max,
            "matching": "exact exclusive runtime assignment",
        },
        "ptr": {"prediction_row": 11, "frames": ptr_entries},
        "var": {"prediction_row": 61, "voxel_size_m": .3, "frames": var_entries},
        "sources": {
            "native_cache": str(NATIVE), "evidence_cache": str(EVIDENCE),
            "raw_detector_root": str(RAW_ROOT), "hsr_readout": str(READOUT),
            "final_predictions": str(FULL_PKL),
        },
    }
    readme = """# HSR / PTR / VAR 真实图片包（scene0599_01）

所有图片来自真实 ScanNet RGB、在线实验缓存和最终预测；没有生成式图片，也没有修改框坐标或尺寸。

## 流程图应使用的图片

- HSR：`01_hsr/...frame_1800_crop.png`。蓝框是原生记录 ID=321 在帧 1800 的真实投影，橙框是同帧独占匹配的 post-NMS proposal；`hsr_identity321_real_history.png` 使用帧 1725/1750/1775/1800 的真实支持值。该案例的历史最大支持从 0.694 更新到 0.726，分数从 0.589 更新到 0.692，帧内排序从第 10 提升到第 8。
- PTR：`02_ptr/ptr_lifted3d_...crop.png` 三张。它们是帧 0/25/50 中与最终 PTR 输出行 11 对应的真实 post-NMS proposal，经 Boxer 提升后的三维框反投影；最后接 `ptr_recovery_3d.png`。
- VAR：`03_var/var_lifted3d_...crop.png` 三张。它们是帧 825/850/875 中与最终 VAR 输出行 61 对应的真实 Top-300 pre-NMS anchor，经 Boxer 提升后的三维框反投影；最后接 `var_recovery_3d.png`。

`04_audit/` 中保留二维 detector 矩形及数值表。旧 VAR 图曾把原分辨率 anchor 框再次按 640×480 放大；本包已经修正，并优先推荐直接显示方法实际使用的三维提升框。

`Voxel consensus` 表示内部体素状态，本身没有对应 RGB 照片，流程图中继续使用简洁体素示意图即可。
"""
    (OUT / "README_CN.md").write_text(readme, encoding="utf-8")
    manifest["files"] = {
        str(path.relative_to(OUT)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(OUT.rglob("*")) if path.is_file()
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(OUT.name) / path.relative_to(OUT))
    print(OUT)
    print(ZIP)


if __name__ == "__main__":
    main()
