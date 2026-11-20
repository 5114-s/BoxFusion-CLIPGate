#!/usr/bin/env python3
"""Export audited real assets for the Auxiliary Candidate Path panel.

Both lanes use the same ScanNet observation (scene0599_01, keyframe 0):

* orange: cached post-NMS proposals lifted by the shared Boxer route;
* gold: cached Top-300 pre-NMS anchors lifted by the same Boxer route.

The compact anchor panel shows the highest-score anchors that did not survive
post-NMS.  This makes the two evidence levels legible in a small method figure.
The archive also includes all-candidate audit panels.  No coordinates are
edited and no generative image is used.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SCENE = "scene0599_01"
FRAME = 0
OUT = ROOT / "figure_assets/auxiliary_candidate_path_verified_real_scene0599_kf0000_20261107"
ZIP = OUT.with_suffix(".zip")
RGB = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color" / f"{FRAME}.jpg"
RAW = ROOT / "reports/scannet_seedless_full100_20260911/raw" / SCENE / f"raw_{FRAME:06d}.npz"
EVIDENCE = (
    ROOT
    / "development/final_controls/experiments/mvsr_unified_samepass_20261010/evidence"
    / f"{SCENE}.npz"
)
NATIVE = (
    ROOT
    / "development/final_controls/experiments/mvsr_multiview_audit_20261012/native_prefix"
    / f"{SCENE}.npz"
)

ORANGE = "#F06418"
GOLD = "#E3A000"
INK = "#142033"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cuboid_edges(box: np.ndarray) -> np.ndarray:
    """Recover the 12 cuboid edges without assuming stored corner order."""
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
        codes = list(itertools.product((0, 1), repeat=3))
        generated = np.asarray([points[0] + np.asarray(code) @ basis for code in codes])
        nearest = np.argmin(
            np.linalg.norm(generated[:, None] - points[None], axis=2), axis=1
        )
        if len(set(nearest.tolist())) != 8:
            continue
        reconstruction = float(
            np.max(np.linalg.norm(generated - points[nearest], axis=1))
        )
        diagonal_error = float(np.linalg.norm(basis.sum(axis=0) - diagonal))
        cost = reconstruction + diagonal_error
        if cost < best_cost:
            best_cost = cost
            best = dict(zip(codes, nearest.tolist()))
    if best is None or best_cost > 1e-4:
        raise ValueError(f"cannot recover cuboid topology: {best_cost}")
    edges = {
        tuple(sorted((best[first], best[second])))
        for first in best
        for second in best
        if sum(a != b for a, b in zip(first, second)) == 1
    }
    if len(edges) != 12:
        raise AssertionError(len(edges))
    return np.asarray(sorted(edges), dtype=int)


def frame_slices(lengths: np.ndarray, index: int) -> slice:
    offsets = np.r_[0, np.cumsum(lengths)]
    return slice(int(offsets[index]), int(offsets[index + 1]))


def project_world(corners: np.ndarray, pose: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inverse = np.linalg.inv(pose)
    homogeneous = np.concatenate(
        [corners, np.ones(corners.shape[:-1] + (1,), dtype=float)], axis=-1
    )
    camera = homogeneous @ inverse.T
    z = camera[..., 2]
    uv = np.empty(camera.shape[:-1] + (2,), dtype=float)
    uv[..., 0] = intrinsic[0, 0] * camera[..., 0] / z + intrinsic[0, 2]
    uv[..., 1] = intrinsic[1, 1] * camera[..., 1] / z + intrinsic[1, 2]
    return uv, z


def bounds(uv: np.ndarray) -> np.ndarray:
    return np.concatenate([uv.min(axis=1), uv.max(axis=1)], axis=1)


def iou2d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    low = np.maximum(left[:, :2], right[:, :2])
    high = np.minimum(left[:, 2:], right[:, 2:])
    intersection = np.maximum(0.0, high - low).prod(axis=1)
    area_left = np.maximum(0.0, left[:, 2] - left[:, 0]) * np.maximum(0.0, left[:, 3] - left[:, 1])
    area_right = np.maximum(0.0, right[:, 2] - right[:, 0]) * np.maximum(0.0, right[:, 3] - right[:, 1])
    return intersection / np.maximum(area_left + area_right - intersection, 1e-12)


def greedy_diverse_rows(
    boxes: np.ndarray,
    scores: np.ndarray,
    candidate_rows: np.ndarray,
    *,
    count: int,
    overlap_limit: float,
) -> np.ndarray:
    """Select high-score rows while avoiding an unreadable stack of duplicates."""
    selected: list[int] = []
    for row in candidate_rows[np.argsort(-scores[candidate_rows], kind="stable")]:
        box = boxes[int(row)]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area < 0.004 * 1296 * 968 or area > 0.72 * 1296 * 968:
            continue
        if selected:
            left = np.repeat(box[None], len(selected), axis=0)
            if float(np.max(iou2d(left, boxes[selected]))) > overlap_limit:
                continue
        selected.append(int(row))
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=int)


def scale_points(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    value = np.asarray(points, dtype=float).copy()
    value[..., 0] *= sx
    value[..., 1] *= sy
    return value


def scale_boxes(boxes: np.ndarray, sx: float, sy: float) -> np.ndarray:
    value = np.asarray(boxes, dtype=float).copy()
    value[:, [0, 2]] *= sx
    value[:, [1, 3]] *= sy
    return value


def common_crop_box(image: Image.Image, boxes: np.ndarray, pad: float = 0.04) -> tuple[int, int, int, int]:
    clipped = np.asarray(boxes, dtype=float).copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, image.width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, image.height)
    low = clipped[:, :2].min(axis=0)
    high = clipped[:, 2:].max(axis=0)
    span = high - low
    low -= max(span) * pad
    high += max(span) * pad
    low = np.maximum(low, 0)
    high = np.minimum(high, [image.width, image.height])
    cx, cy = (low + high) / 2
    width, height = high - low
    target = 1.42
    if width / max(height, 1) < target:
        width = height * target
    else:
        height = width / target
    return (
        round(max(0, cx - width / 2)), round(max(0, cy - height / 2)),
        round(min(image.width, cx + width / 2)), round(min(image.height, cy + height / 2)),
    )


def draw_rectangles(image: Image.Image, boxes: np.ndarray, color: str, *, audit: bool = False) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    width = 2 if audit else max(5, round(image.width / 230))
    halo = width + (2 if audit else 5)
    for box in boxes:
        xy = tuple(map(float, box))
        if not audit:
            draw.rectangle(xy, outline="white", width=halo)
        draw.rectangle(xy, outline=color, width=width)
    return output


def draw_wireframes(image: Image.Image, boxes: np.ndarray, uv: np.ndarray, color: str, *, audit: bool = False) -> Image.Image:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    width = 2 if audit else max(5, round(image.width / 230))
    halo = width + (2 if audit else 5)
    for box, corners in zip(boxes, uv):
        for first, second in cuboid_edges(box):
            segment = [tuple(corners[first]), tuple(corners[second])]
            if not audit:
                draw.line(segment, fill="white", width=halo)
            draw.line(segment, fill=color, width=width)
    return output


def save_full_crop(image: Image.Image, stem: str, crop: tuple[int, int, int, int]) -> list[Path]:
    full = OUT / f"{stem}_full.png"
    cropped = OUT / f"{stem}_crop.png"
    image.save(full)
    image.crop(crop).save(cropped)
    return [full, cropped]


def contact_sheet(items: list[tuple[str, Path]], destination: Path) -> None:
    tile_w, tile_h, label_h = 680, 440, 54
    sheet = Image.new("RGB", (tile_w * 2, (tile_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        image = ImageOps.contain(image, (tile_w - 28, tile_h - 28), Image.Resampling.LANCZOS)
        col, row = index % 2, index // 2
        x = col * tile_w + (tile_w - image.width) // 2
        y = row * (tile_h + label_h) + (tile_h - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((col * tile_w + 20, row * (tile_h + label_h) + tile_h + 7), label, fill=INK, font=font)
    sheet.save(destination)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    evidence = np.load(EVIDENCE, allow_pickle=False)
    native = np.load(NATIVE, allow_pickle=False)
    raw = np.load(RAW, allow_pickle=False)
    frame_index = int(np.where(evidence["frame_ids"] == FRAME)[0][0])
    native_index = int(np.where(native["frame_ids"] == FRAME)[0][0])
    proposal_slice = frame_slices(evidence["proposal_lengths"], frame_index)
    anchor_slice = frame_slices(evidence["anchor_lengths"], frame_index)

    proposal_ids = evidence["proposal_ids"][proposal_slice].astype(int)
    proposal_scores = evidence["proposal_scores"][proposal_slice].astype(float)
    proposal_2d = evidence["proposal_boxes_2d"][proposal_slice].astype(float)
    proposal_corners = evidence["proposal_corners"][proposal_slice].astype(float)
    anchor_ids = evidence["anchor_ids"][anchor_slice].astype(int)
    anchor_scores = evidence["anchor_scores"][anchor_slice].astype(float)
    anchor_corners = evidence["anchor_corners"][anchor_slice].astype(float)
    anchor_2d = raw["boxes"][anchor_ids].astype(float)

    pose = native["camera_to_world"][native_index].astype(float)
    intrinsic = native["intrinsic"][native_index].astype(float)
    proposal_uv, proposal_z = project_world(proposal_corners, pose, intrinsic)
    anchor_uv, anchor_z = project_world(anchor_corners, pose, intrinsic)

    # The compact proposal lane keeps the four highest-confidence post-NMS rows.
    proposal_order = np.argsort(-proposal_scores, kind="stable")
    proposal_compact = proposal_order[:4]

    # The compact anchor lane exposes the strongest pre-NMS alternatives that
    # were suppressed by NMS.  They are members of the actual Top-300 state.
    suppressed = np.where(~np.isin(anchor_ids, proposal_ids))[0]
    anchor_compact = greedy_diverse_rows(
        anchor_2d, anchor_scores, suppressed, count=6, overlap_limit=.42
    )

    image = Image.open(RGB).convert("RGB")
    sx, sy = image.width / 640.0, image.height / 480.0
    proposal_uv_scaled = scale_points(proposal_uv, sx, sy)
    anchor_uv_scaled = scale_points(anchor_uv, sx, sy)
    proposal_2d_scaled = scale_boxes(proposal_2d, sx, sy)
    # raw anchor rectangles are already stored in the original RGB resolution.
    # The old pack incorrectly scaled them a second time.
    anchor_2d_scaled = anchor_2d.copy()
    anchor_2d_camera = scale_boxes(anchor_2d, 640.0 / image.width, 480.0 / image.height)

    crop_boxes = np.concatenate([
        bounds(proposal_uv_scaled[proposal_compact]),
        bounds(anchor_uv_scaled[anchor_compact]),
    ])
    crop = common_crop_box(image, crop_boxes)

    files: list[Path] = []
    p2 = draw_rectangles(image, proposal_2d_scaled[proposal_compact], ORANGE)
    files += save_full_crop(p2, "01_post_nms_proposals_2d_compact", crop)
    p3 = draw_wireframes(
        image, proposal_corners[proposal_compact], proposal_uv_scaled[proposal_compact], ORANGE
    )
    files += save_full_crop(p3, "02_lifted_post_nms_proposals_3d_compact", crop)
    a2 = draw_rectangles(image, anchor_2d_scaled[anchor_compact], GOLD)
    files += save_full_crop(a2, "03_top300_prenms_anchors_2d_compact", crop)
    a3 = draw_wireframes(
        image, anchor_corners[anchor_compact], anchor_uv_scaled[anchor_compact], GOLD
    )
    files += save_full_crop(a3, "04_lifted_top300_anchors_3d_compact", crop)

    # Full-state audit panels.  Thin lines are intentional because these files
    # show all 13 proposals or all 300 anchors and are not figure assets.
    draw_rectangles(image, proposal_2d_scaled, ORANGE, audit=True).save(
        OUT / "audit_post_nms_proposals_2d_all13.png"
    )
    draw_wireframes(image, proposal_corners, proposal_uv_scaled, ORANGE, audit=True).save(
        OUT / "audit_lifted_post_nms_proposals_3d_all13.png"
    )
    draw_rectangles(image, anchor_2d_scaled, GOLD, audit=True).save(
        OUT / "audit_top300_prenms_anchors_2d_all300.png"
    )
    draw_wireframes(image, anchor_corners, anchor_uv_scaled, GOLD, audit=True).save(
        OUT / "audit_lifted_top300_anchors_3d_all300.png"
    )
    image.save(OUT / f"rgb_frame_{FRAME:04d}.png")

    contact_sheet([
        ("post-NMS inputs (compact)", OUT / "01_post_nms_proposals_2d_compact_crop.png"),
        ("lifted proposals P_t (recommended)", OUT / "02_lifted_post_nms_proposals_3d_compact_crop.png"),
        ("pre-NMS Top-300 inputs (subset)", OUT / "03_top300_prenms_anchors_2d_compact_crop.png"),
        ("lifted anchors A_t (recommended)", OUT / "04_lifted_top300_anchors_3d_compact_crop.png"),
    ], OUT / "00_contact_sheet.png")

    proposal_projection_iou = iou2d(bounds(proposal_uv), proposal_2d)
    anchor_projection_iou = iou2d(bounds(anchor_uv), anchor_2d_camera)
    manifest = {
        "schema": "carr3d.auxiliary_candidate_path.real_assets.v1",
        "scene": SCENE,
        "keyframe": FRAME,
        "same_observation_for_both_lanes": True,
        "counts": {
            "post_nms_proposals": int(len(proposal_ids)),
            "top300_prenms_anchors": int(len(anchor_ids)),
            "compact_proposals_shown": int(len(proposal_compact)),
            "compact_suppressed_anchors_shown": int(len(anchor_compact)),
        },
        "compact_selection": {
            "proposal_rows": proposal_compact.tolist(),
            "proposal_ids": proposal_ids[proposal_compact].tolist(),
            "proposal_scores": proposal_scores[proposal_compact].tolist(),
            "anchor_rows": anchor_compact.tolist(),
            "anchor_ids": anchor_ids[anchor_compact].tolist(),
            "anchor_scores": anchor_scores[anchor_compact].tolist(),
            "anchor_rule": "score-ordered, spatially diverse Top-300 anchors not present in post-NMS proposal IDs",
        },
        "projection_audit": {
            "proposal_median_2d_iou": float(np.median(proposal_projection_iou)),
            "compact_proposal_2d_iou": proposal_projection_iou[proposal_compact].tolist(),
            "anchor_median_2d_iou": float(np.median(anchor_projection_iou)),
            "compact_anchor_2d_iou": anchor_projection_iou[anchor_compact].tolist(),
            "proposal_positive_depth": bool(np.all(proposal_z > 0)),
            "anchor_positive_depth": bool(np.all(anchor_z > 0)),
        },
        "sources": {
            "rgb": str(RGB),
            "rgb_sha256": sha256(RGB),
            "evidence_cache": str(EVIDENCE),
            "evidence_cache_sha256": sha256(EVIDENCE),
            "raw_detector_cache": str(RAW),
            "raw_detector_cache_sha256": sha256(RAW),
            "native_camera_cache": str(NATIVE),
            "native_camera_cache_sha256": sha256(NATIVE),
        },
        "geometry_edited": False,
        "generative_images_used": False,
        "note": "Compact panels show exact cached candidates; only the number of displayed rows is reduced for legibility.",
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "README_CN.md").write_text(
        f"""# Auxiliary Candidate Path 真实图片包\n\n"
        f"场景：`{SCENE}`；关键帧：`{FRAME}`。上下两支来自同一 RGB 帧和同一次在线候选状态。\n\n"
        "## 流程图推荐图片\n\n"
        "- `02_lifted_post_nms_proposals_3d_compact_crop.png`：真实 post-NMS proposals 经 Boxer 提升后的三维线框，作为 $P_t$。\n"
        "- `04_lifted_top300_anchors_3d_compact_crop.png`：同一帧 Top-300 中未通过 NMS 的高分 anchors 经 Boxer 提升后的三维线框，作为 $A_t$。\n\n"
        "橙色和黄色图并非同一组框。流程图版本只减少显示数量，没有修改任何候选的坐标、尺寸或旋转；`audit_*` 文件保留全部 13 个 proposal 和全部 300 个 anchor。\n"
        """,
        encoding="utf-8",
    )

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(OUT.name) / path.relative_to(OUT))
    print(json.dumps({"folder": str(OUT), "zip": str(ZIP)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
