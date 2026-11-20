#!/usr/bin/env python3
"""Build an audited, non-generative image pack for the CaRR-3D framework.

Every photographic panel comes from the archived scene0599_01 RGB-D stream.
Every 2D overlay is read from the exact cached Native/Evidence frames used by
the online experiments.  Every 3D result panel uses archived prediction
coordinates without geometric editing.  No generative image tool is used.
"""
from __future__ import annotations

import csv
import hashlib
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
OUT = ROOT / "figure_assets/carr3d_framework_verified_real_pack_scene0599_20261107"
ZIP = OUT.with_suffix(".zip")
RGB_ROOT = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/color"
DEPTH_ROOT = ROOT / "data/scannet_val_rgbfix" / SCENE / "frames/depth"
NATIVE_CACHE = ROOT / "development/final_controls/experiments/mvsr_multiview_audit_20261012/native_prefix" / f"{SCENE}.npz"
EVIDENCE_CACHE = ROOT / "development/final_controls/experiments/mvsr_unified_samepass_20261010/evidence" / f"{SCENE}.npz"
RAW_ROOT = ROOT / "reports/scannet_seedless_full100_20260911/raw" / SCENE
SOURCE_SNAPSHOT = ROOT / "development/mvsr_multiview_audit_20261012_source"
RUN = ROOT / "reports/official_github_boxfusion_vs_final_recar_public_cli_coldcache_nowarmup100_20261009"
RECEIPT = RUN / "receipts/recar" / SCENE
NATIVE_PKL = RECEIPT / "native" / f"{SCENE}_boxes.pkl"
FULL_PKL = RECEIPT / "recar" / f"{SCENE}_boxes.pkl"
BOXER_LOG = RECEIPT / "native_boxer_diagnostics" / f"{SCENE}_boxer_lifting.jsonl"
HSR_READOUT = ROOT / "reports/scannet_hcsr_current_zero_vs_max_20261029/frame_readouts" / f"{SCENE}.npz"
OLD_REAL = ROOT / "figure_assets/carr3d_framework_scene0599_real_20261107"
READOUT = ROOT / "figure_assets/online_detection_readout_real_dual_branch_20261107"

BLUE = "#236FB6"
ORANGE = "#F07820"
GOLD = "#E0A000"

sys.path.insert(0, str(SOURCE_SNAPSHOT))
from boxfusion.native_reliability_reranker import _project_with_visibility  # noqa: E402
from boxfusion.online_candidate_map import _aabb_iou, _iou2d_matrix  # noqa: E402
from tools import evaluate_scannet_mechanism_controls as ev  # noqa: E402

core = ev.audit.core


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path) -> list[tuple]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload[0] if len(payload) == 1 and isinstance(payload[0], list) else payload


def scale_rect(rect: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(rect, dtype=float).copy()
    out[[0, 2]] *= width / 640.0
    out[[1, 3]] *= height / 480.0
    return out


def crop_around(image: Image.Image, boxes: list[np.ndarray], ratio: float = 1.35) -> Image.Image:
    if not boxes:
        return image.copy()
    values = np.asarray(boxes, dtype=float)
    x1, y1 = values[:, :2].min(0)
    x2, y2 = values[:, 2:].max(0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 80), max(y2 - y1, 80)
    bw, bh = bw * ratio, bh * ratio
    target = 1.55
    if bw / bh < target:
        bw = bh * target
    else:
        bh = bw / target
    left, top = max(0, cx - bw / 2), max(0, cy - bh / 2)
    right, bottom = min(image.width, cx + bw / 2), min(image.height, cy + bh / 2)
    return image.crop((round(left), round(top), round(right), round(bottom)))


def overlay(frame: int, boxes: list[np.ndarray], colors: list[str], stem: Path) -> tuple[Path, Path]:
    image = Image.open(RGB_ROOT / f"{frame}.jpg").convert("RGB")
    scaled = [scale_rect(box, image.width, image.height) for box in boxes]
    draw = ImageDraw.Draw(image)
    width = max(4, round(image.width / 260))
    for box, color in zip(scaled, colors):
        draw.rectangle(tuple(map(float, box)), outline=color, width=width)
    full = stem.with_name(stem.name + "_full.png")
    crop = stem.with_name(stem.name + "_crop.png")
    image.save(full)
    crop_around(image, scaled).save(crop)
    return full, crop


def depth_render(frame: int, destination: Path) -> None:
    depth = np.asarray(Image.open(DEPTH_ROOT / f"{frame}.png"), dtype=float)
    valid = depth > 0
    lo, hi = np.percentile(depth[valid], [4, 96]) if np.any(valid) else (0.0, 1.0)
    norm = np.clip((depth - lo) / max(hi - lo, 1.0), 0.0, 1.0)
    out = (255 * (1.0 - norm)).astype(np.uint8)
    out[~valid] = 245
    Image.fromarray(out, mode="L").save(destination)


def exact_assignments(native, evidence):
    projected_rows, projected_boxes = [], []
    projections = {}
    for row, corners in enumerate(native.corners):
        value = _project_with_visibility(
            corners, native.camera_to_world, native.intrinsic,
            native.width, native.height,
        )
        if value is None:
            continue
        projected_rows.append(row)
        projected_boxes.append(value[0])
        projections[row] = value[0]
    assigned = np.full(len(native.ids), -1, dtype=np.int64)
    frame_iou = np.zeros(len(native.ids), dtype=float)
    if projected_boxes and len(evidence.proposal_boxes_2d):
        overlap = _iou2d_matrix(np.asarray(projected_boxes), evidence.proposal_boxes_2d)
        pairs = [
            (float(overlap[r, c]), r, c)
            for r, c in zip(*np.where(overlap >= 0.1))
        ]
        pairs.sort(key=lambda value: (-value[0], value[1], value[2]))
        used_rows, used_proposals = set(), set()
        for value, projected_row, proposal_row in pairs:
            if projected_row in used_rows or proposal_row in used_proposals:
                continue
            used_rows.add(projected_row)
            used_proposals.add(proposal_row)
            native_row = projected_rows[projected_row]
            assigned[native_row] = proposal_row
            frame_iou[native_row] = value
    return assigned, frame_iou, projections


def aabb_iou(left: np.ndarray, right: np.ndarray) -> float:
    return float(_aabb_iou(np.asarray(left), np.asarray(right)))


def raw_anchor_box(frame: int, anchor_id: int) -> np.ndarray:
    """Return the original detector rectangle for a cached anchor identity."""
    with np.load(RAW_ROOT / f"raw_{frame:06d}.npz") as raw:
        return np.asarray(raw["boxes"][int(anchor_id)], dtype=float)


def contact_sheet(items: list[tuple[str, Path]], destination: Path, columns: int = 3) -> None:
    thumb_w, thumb_h, label_h = 420, 265, 42
    rows = (len(items) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    for index, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        fitted = ImageOps.contain(image, (thumb_w - 20, thumb_h - 20), Image.Resampling.LANCZOS)
        col, row = index % columns, index // columns
        x = col * thumb_w + (thumb_w - fitted.width) // 2
        y = row * (thumb_h + label_h) + (thumb_h - fitted.height) // 2
        sheet.paste(fitted, (x, y))
        draw.text((col * thumb_w + 12, row * (thumb_h + label_h) + thumb_h + 5), label, fill="#142033", font=font)
    sheet.save(destination)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for folder in [
        "01_input_stream", "02_native_mapping", "03_auxiliary_candidates",
        "04_hsr_exact_identity20", "05_ptr_exact_target",
        "06_var_exact_target", "07_online_readout", "08_audit",
    ]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)

    natives = core.load_native(NATIVE_CACHE)
    evidences = core.load_evidence(EVIDENCE_CACHE)
    by_frame = {int(frame): (native, evidence) for (frame, native), (_, evidence) in zip(natives, evidences)}
    full_rows = load_rows(FULL_PKL)
    ptr_target = np.asarray(full_rows[11][1], dtype=float)
    var_target = np.asarray(full_rows[61][1], dtype=float)
    audit: dict[str, object] = {
        "schema": "carr3d.framework.verified_real_pack.v1",
        "scene": SCENE,
        "generative_images_used": False,
        "prediction_geometry_edited": False,
        "source_files": {
            "native_cache": str(NATIVE_CACHE),
            "evidence_cache": str(EVIDENCE_CACHE),
            "raw_detector_root": str(RAW_ROOT),
            "native_predictions": str(NATIVE_PKL),
            "full_predictions": str(FULL_PKL),
            "boxer_log": str(BOXER_LOG),
            "hsr_readout": str(HSR_READOUT),
        },
    }

    # 1. Exact RGB-D observations.
    preview_items: list[tuple[str, Path]] = []
    for frame in (0, 450, 900):
        rgb_out = OUT / "01_input_stream" / f"rgb_frame_{frame:04d}.png"
        depth_out = OUT / "01_input_stream" / f"depth_frame_{frame:04d}.png"
        Image.open(RGB_ROOT / f"{frame}.jpg").convert("RGB").save(rgb_out)
        depth_render(frame, depth_out)
        preview_items.extend([(f"RGB {frame}", rgb_out), (f"Depth {frame}", depth_out)])

    # 2. Native per-frame 2D inputs and actual Boxer 3D projections.
    boxer_rows = {int(json.loads(line)["frame_id"]): json.loads(line) for line in BOXER_LOG.open()}
    boxer_frame = 25
    row = boxer_rows[boxer_frame]
    input_boxes = np.asarray(row["input_boxes_xyxy"], dtype=float)[:6]
    input_full, input_crop = overlay(
        boxer_frame, list(input_boxes), [BLUE] * len(input_boxes),
        OUT / "02_native_mapping" / f"native_2d_inputs_frame_{boxer_frame:04d}",
    )
    # Boxer diagnostic uses [x1, x2, y1, y2] in the original RGB resolution.
    rgb = Image.open(RGB_ROOT / f"{boxer_frame}.jpg").convert("RGB")
    projected = np.asarray(row["boxer_boxes_xxyy"], dtype=float)[:6]
    projected_xyxy = projected[:, [0, 2, 1, 3]]
    projected_xyxy[:, [0, 2]] *= 640.0 / rgb.width
    projected_xyxy[:, [1, 3]] *= 480.0 / rgb.height
    lift_full, lift_crop = overlay(
        boxer_frame, list(projected_xyxy), [BLUE] * len(projected_xyxy),
        OUT / "02_native_mapping" / f"native_boxer_3d_projection_frame_{boxer_frame:04d}",
    )
    for name in ("previous_map.png", "native_map.png"):
        shutil.copy2(OLD_REAL / name, OUT / "02_native_mapping" / name)
    preview_items.extend([
        ("Native 2D inputs", input_crop), ("Boxer 3D projection", lift_crop),
        ("Native map", OUT / "02_native_mapping/native_map.png"),
    ])

    # 3. Exact cached post-NMS proposals and Top-300 pre-NMS anchors.
    prop_frame = 50
    _, evidence = by_frame[prop_frame]
    p_order = np.argsort(-evidence.proposal_scores, kind="stable")[:6]
    prop_full, prop_crop = overlay(
        prop_frame, [evidence.proposal_boxes_2d[i] for i in p_order],
        [ORANGE] * len(p_order), OUT / "03_auxiliary_candidates" / f"post_nms_proposals_frame_{prop_frame:04d}",
    )
    anchor_frame = 850
    _, evidence = by_frame[anchor_frame]
    a_order = np.argsort(-evidence.anchor_scores, kind="stable")[:8]
    anchor_full, anchor_crop = overlay(
        anchor_frame, [raw_anchor_box(anchor_frame, evidence.anchor_ids[i]) for i in a_order],
        [GOLD] * len(a_order), OUT / "03_auxiliary_candidates" / f"top300_anchors_frame_{anchor_frame:04d}",
    )
    preview_items.extend([("Post-NMS proposals", prop_crop), ("Pre-NMS anchors", anchor_crop)])

    # 4. HSR: same persistent native identity, exact runtime exclusive matches.
    hsr_rows = []
    for frame in (150, 175, 225):
        native, evidence = by_frame[frame]
        assigned, frame_iou, projections = exact_assignments(native, evidence)
        locations = np.where(native.ids == 20)[0]
        if len(locations) != 1:
            raise RuntimeError(f"identity 20 missing or duplicated at frame {frame}")
        native_row = int(locations[0])
        proposal_row = int(assigned[native_row])
        if proposal_row < 0:
            raise RuntimeError(f"identity 20 unmatched at frame {frame}")
        full, crop = overlay(
            frame,
            [projections[native_row], evidence.proposal_boxes_2d[proposal_row]],
            [BLUE, ORANGE], OUT / "04_hsr_exact_identity20" / f"hsr_id20_match_frame_{frame:04d}",
        )
        hsr_rows.append({
            "frame": frame,
            "native_identity": 20,
            "native_row": native_row,
            "proposal_row": proposal_row,
            "proposal_id": int(evidence.proposal_ids[proposal_row]),
            "proposal_score": float(evidence.proposal_scores[proposal_row]),
            "iou2d": float(frame_iou[native_row]),
        })
        preview_items.append((f"HSR id20 · {frame}", crop))
    audit["hsr"] = {"identity": 20, "frames": hsr_rows, "matching": "exact exclusive runtime assignment"}

    # 5. PTR: post-NMS proposals selected by 3D overlap with the unchanged
    # archived PTR output (full prediction row 11).
    ptr_rows = []
    for frame in (0, 25, 50):
        _, evidence = by_frame[frame]
        overlaps = np.asarray([aabb_iou(ptr_target, box) for box in evidence.proposal_corners])
        proposal_row = int(np.argmax(overlaps))
        full, crop = overlay(
            frame, [evidence.proposal_boxes_2d[proposal_row]], [ORANGE],
            OUT / "05_ptr_exact_target" / f"ptr_proposal_frame_{frame:04d}",
        )
        ptr_rows.append({
            "frame": frame,
            "proposal_row": proposal_row,
            "proposal_id": int(evidence.proposal_ids[proposal_row]),
            "proposal_score": float(evidence.proposal_scores[proposal_row]),
            "aabb_iou_to_ptr_output": float(overlaps[proposal_row]),
        })
        preview_items.append((f"PTR proposal · {frame}", crop))
    shutil.copy2(OLD_REAL / "ptr_recovery.png", OUT / "05_ptr_exact_target/ptr_recovery_3d.png")
    preview_items.append(("PTR recovery 3D", OUT / "05_ptr_exact_target/ptr_recovery_3d.png"))
    audit["ptr"] = {"prediction_row": 11, "frames": ptr_rows}

    # 6. VAR: pre-NMS anchors selected by 3D overlap with the unchanged
    # archived VAR output (full prediction row 61).
    var_rows = []
    for frame in (825, 850, 875):
        _, evidence = by_frame[frame]
        overlaps = np.asarray([aabb_iou(var_target, box) for box in evidence.anchor_corners])
        anchor_row = int(np.argmax(overlaps))
        full, crop = overlay(
            frame, [raw_anchor_box(frame, evidence.anchor_ids[anchor_row])], [GOLD],
            OUT / "06_var_exact_target" / f"var_anchor_frame_{frame:04d}",
        )
        var_rows.append({
            "frame": frame,
            "anchor_row": anchor_row,
            "anchor_id": int(evidence.anchor_ids[anchor_row]),
            "anchor_score": float(evidence.anchor_scores[anchor_row]),
            "aabb_iou_to_var_output": float(overlaps[anchor_row]),
        })
        preview_items.append((f"VAR anchor · {frame}", crop))
    shutil.copy2(OLD_REAL / "var_recovery.png", OUT / "06_var_exact_target/var_recovery_3d.png")
    preview_items.append(("VAR recovery 3D", OUT / "06_var_exact_target/var_recovery_3d.png"))
    audit["var"] = {"prediction_row": 61, "frames": var_rows}

    # 7. Preserve the requested readout files byte-for-byte.
    for name in [
        "online_detection_readout_dual_white.png",
        "online_detection_readout_dual_transparent.png",
        "online_detection_readout_dual_audit.png",
        "source_manifest.json",
    ]:
        shutil.copy2(READOUT / name, OUT / "07_online_readout" / name)
    preview_items.append(("Exact online readout", OUT / "07_online_readout/online_detection_readout_dual_white.png"))

    # Audit tables and human-readable mapping.
    with (OUT / "08_audit/hsr_exact_matches.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=hsr_rows[0].keys())
        writer.writeheader(); writer.writerows(hsr_rows)
    with (OUT / "08_audit/ptr_exact_matches.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ptr_rows[0].keys())
        writer.writeheader(); writer.writerows(ptr_rows)
    with (OUT / "08_audit/var_exact_matches.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=var_rows[0].keys())
        writer.writeheader(); writer.writerows(var_rows)

    contact_sheet(preview_items, OUT / "00_contact_sheet.png", columns=4)
    readme = f"""# CaRR-3D 流程图真实图片包（{SCENE}）

本压缩包不包含任何生成式图片。所有 RGB/深度图来自真实 ScanNet 帧；候选框来自实验使用的缓存；三维框来自已归档预测坐标，未修改几何。

## 流程图槽位

- `01_input_stream/`：真实 RGB-D 输入帧。
- `02_native_mapping/`：原生二维输入、Boxer 三维投影和原生地图。
- `03_auxiliary_candidates/`：实验缓存中的 post-NMS proposals 与 Top-300 pre-NMS anchors。
- `04_hsr_exact_identity20/`：同一原生身份 `ID=20` 在帧 150/175/225 的精确独占匹配。蓝框为原生记录投影，橙框为实际匹配 proposal。
- `05_ptr_exact_target/`：与最终 PTR 输出行 11 对应的 post-NMS proposal 证据，帧 0/25/50。
- `06_var_exact_target/`：与最终 VAR 输出行 61 对应的 pre-NMS anchor 证据，帧 825/850/875。
- `07_online_readout/`：指定的最终结果原文件，按字节复制。
- `08_audit/`：候选 ID、分数和重叠度。

`*_full.png` 保留完整 RGB 帧，`*_crop.png` 是同一图像的紧裁剪版本，框坐标未改变。
"""
    (OUT / "README_CN.md").write_text(readme, encoding="utf-8")

    audit["files"] = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            audit["files"][str(path.relative_to(OUT))] = {
                "sha256": sha256(path), "bytes": path.stat().st_size,
            }
    (OUT / "manifest.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                archive.write(path, Path(OUT.name) / path.relative_to(OUT))
    print(json.dumps({"folder": str(OUT), "zip": str(ZIP), "files": len(list(OUT.rglob('*')))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
