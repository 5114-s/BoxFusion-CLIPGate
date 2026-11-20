#!/usr/bin/env python3
"""CoDA-frequency-group semantic detection audit on ScanNet-100.

This script never reruns the 3D detector.  It reads the frozen final prediction
files, reconstructs ScanNet200 boxes from the licensed ScanNet annotations, and
assigns one of the top-60 CoDA categories to every prediction with frozen CLIP.

The frozen system was not trained with CoDA's base-only supervision.  Therefore
the two groups are frequent-10/tail-50 diagnostics, not a valid seen/unseen
open-vocabulary claim.  The scene set is also the paper's ScanNet-100 subset,
not CoDA's complete ScanNet validation set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from paper_eval_core import best_view, corners_from_center_size
from true_fusion_audit_core import class_agnostic_ap

DEFAULT_PRED_ROOT = (
    ROOT / "development/final_controls/results/"
    "scannet_recar3d_key_control_arms"
)
DEFAULT_OUT = ROOT / "reports/scannet_base_novel_final_20261010"
DEFAULT_SCANS = Path("/extra/ZhaoX/scannet_data/scans")
DEFAULT_META = Path("/extra/ZhaoX/scannet_data/meta_data")
DEFAULT_FRAMES = ROOT / "upstream_clean/scannet_readme_frames"
THRESHOLDS = (0.15, 0.25, 0.50)
STUFF = {"wall", "floor", "ceiling"}

# CoDA's released ScanNet commands use exactly these ten raw ScanNet200 IDs as
# training/base categories.  The names also equal the ten most frequent object
# categories in the local ScanNet training annotations.
CODA_BASE_IDS = {2, 4, 5, 7, 13, 15, 16, 22, 56, 1163}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_label_rows(mapping: Path) -> list[dict[str, str]]:
    with mapping.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_split(meta_root: Path, scans_root: Path) -> dict:
    mapping = meta_root / "scannetv2-labels.combined.tsv"
    rows = load_label_rows(mapping)
    raw_to_category = {row["raw_category"]: row["category"] for row in rows}
    category_to_ids: dict[str, set[int]] = {}
    for row in rows:
        category_to_ids.setdefault(row["category"], set()).add(int(row["id"]))

    train_scenes = (meta_root / "scannetv2_train.txt").read_text().split()
    counts: Counter[str] = Counter()
    missing = []
    for scene in train_scenes:
        annotation = scans_root / scene / f"{scene}.aggregation.json"
        if not annotation.exists():
            missing.append(scene)
            continue
        groups = json.loads(annotation.read_text())["segGroups"]
        for group in groups:
            category = raw_to_category.get(group["label"])
            if category is not None and category not in STUFF:
                counts[category] += 1
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} ScanNet training annotations")

    ranked = counts.most_common(60)
    base_names = [name for name, _ in ranked[:10]]
    novel_names = [name for name, _ in ranked[10:60]]
    base_ids = set().union(*(category_to_ids[name] for name in base_names))
    # A category can have aliases/duplicate raw IDs.  CoDA's command lists one
    # canonical raw ID per base name, so compare through the canonical mapping.
    canonical = {
        name: min(category_to_ids[name], key=lambda value: (
            0 if any(int(r["id"]) == value and r["raw_category"] == name
                     for r in rows) else 1,
            value,
        ))
        for name in base_names
    }
    if set(canonical.values()) != CODA_BASE_IDS:
        raise AssertionError(
            f"local top-10 does not reproduce CoDA base IDs: {canonical}"
        )
    return {
        "schema": "boxfusion.scannet100.coda_frequency_groups.v2",
        "definition": (
            "Top 10 object categories by instance frequency in the 1201-scene "
            "ScanNet training split form frequent-10; ranks 11-60 form tail-50. "
            "These groups are not seen/unseen without base-only training."
        ),
        "claim_boundary": (
            "Frequency-group diagnostic only; not a base/novel open-vocabulary "
            "generalization result."
        ),
        "base_names": base_names,
        "novel_names": novel_names,
        "all_names": base_names + novel_names,
        "base_canonical_raw_ids": canonical,
        "base_released_command_ids": sorted(CODA_BASE_IDS),
        "training_instance_counts": {name: int(counts[name])
                                     for name, _ in ranked},
        "training_scene_count": len(train_scenes),
        "mapping_sha256": sha256(mapping),
        "train_split_sha256": sha256(meta_root / "scannetv2_train.txt"),
    }


def alignment(scene_root: Path, scene: str) -> np.ndarray:
    for line in (scene_root / scene / f"{scene}.txt").read_text().splitlines():
        if line.startswith("axisAlignment"):
            return np.fromstring(line.split("=", 1)[1], sep=" ").reshape(4, 4)
    raise ValueError(f"axisAlignment missing: {scene}")


def scene_gt(scene_root: Path, meta_root: Path, scene: str,
             names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    from plyfile import PlyData

    root = scene_root / scene
    mapping = load_label_rows(meta_root / "scannetv2-labels.combined.tsv")
    raw_to_category = {row["raw_category"]: row["category"] for row in mapping}
    name_to_index = {name: index for index, name in enumerate(names)}

    ply_path = root / f"{scene}_vh_clean_2.ply"
    seg_path = root / f"{scene}_vh_clean_2.0.010000.segs.json"
    agg_path = root / f"{scene}.aggregation.json"
    vertex = PlyData.read(str(ply_path))["vertex"].data
    xyz = np.c_[vertex["x"], vertex["y"], vertex["z"]].astype(np.float64)
    transform = alignment(scene_root, scene)
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    segments = np.asarray(json.loads(seg_path.read_text())["segIndices"],
                          dtype=np.int64)
    if len(segments) != len(xyz):
        raise ValueError(f"vertex/segment length mismatch: {scene}")

    boxes, classes = [], []
    for group in json.loads(agg_path.read_text())["segGroups"]:
        category = raw_to_category.get(group["label"])
        if category not in name_to_index:
            continue
        points = xyz[np.isin(segments, np.asarray(group["segments"]))]
        if not len(points):
            continue
        lo, hi = points.min(0), points.max(0)
        boxes.append(np.r_[(lo + hi) / 2.0, hi - lo])
        classes.append(name_to_index[category])
    center_size = np.asarray(boxes, np.float64).reshape(-1, 6)
    provenance = {
        "aggregation_sha256": sha256(agg_path),
        "segments_sha256": sha256(seg_path),
        "mesh_sha256": sha256(ply_path),
        "metadata_sha256": sha256(root / f"{scene}.txt"),
    }
    return corners_from_center_size(center_size), np.asarray(classes, int), provenance


def validate_gt_export(scene_root: Path, meta_root: Path, scene: str) -> float:
    """Reproduce the existing NYU18 exporter for an independent geometry check."""
    from plyfile import PlyData

    nyu_ids = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
               33, 34, 36, 39}
    rows = load_label_rows(meta_root / "scannetv2-labels.combined.tsv")
    raw_to_nyu = {row["raw_category"]: int(row["nyu40id"]) for row in rows}
    root = scene_root / scene
    vertex = PlyData.read(str(root / f"{scene}_vh_clean_2.ply"))["vertex"].data
    xyz = np.c_[vertex["x"], vertex["y"], vertex["z"]].astype(np.float64)
    transform = alignment(scene_root, scene)
    xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    segments = np.asarray(json.loads(
        (root / f"{scene}_vh_clean_2.0.010000.segs.json").read_text()
    )["segIndices"])
    generated = []
    for group in json.loads((root / f"{scene}.aggregation.json").read_text())["segGroups"]:
        nyu = raw_to_nyu[group["label"]]
        if nyu not in nyu_ids:
            continue
        points = xyz[np.isin(segments, group["segments"])]
        lo, hi = points.min(0), points.max(0)
        generated.append(np.r_[(lo + hi) / 2.0, hi - lo, nyu])
    generated = np.asarray(generated)
    existing = np.load(ROOT / "evaluation/data_util/scannet_train_detection_data" /
                       f"{scene}_bbox.npy")
    if generated.shape != existing.shape:
        raise AssertionError(f"NYU18 GT shape mismatch: {scene}")
    error = float(np.max(np.abs(generated - existing))) if len(existing) else 0.0
    if error > 1e-5:
        raise AssertionError(f"NYU18 GT geometry mismatch: {scene}: {error}")
    return error


def prepare_gt(scenes: list[str], split: dict, scans_root: Path,
               meta_root: Path, out: Path) -> None:
    cache = out / "gt_cache"
    cache.mkdir(parents=True, exist_ok=True)
    audit = {}
    for index, scene in enumerate(scenes):
        boxes, classes, provenance = scene_gt(
            scans_root, meta_root, scene, split["all_names"]
        )
        np.savez_compressed(cache / f"{scene}.npz", boxes=boxes,
                            classes=classes)
        audit[scene] = {
            "gt_boxes": len(boxes),
            "nyu18_export_max_abs_error": validate_gt_export(
                scans_root, meta_root, scene
            ),
            **provenance,
        }
        print(f"GT {index + 1}/{len(scenes)} {scene}: {len(boxes)}", flush=True)
    write_json(out / "gt_audit.json", audit)


def read_rows(path: Path):
    value = pickle.loads(path.read_bytes())
    # Current factorial artifacts use ``[rows]`` while some historical
    # exporters use ``(rows, diagnostics)``.  A bare rows list begins with a
    # three-field prediction tuple and must not be unwrapped.
    if isinstance(value, tuple):
        return value[0]
    if (isinstance(value, list) and len(value) == 1
            and isinstance(value[0], list)):
        return value[0]
    return value


def keyframes(frames_root: Path, scene: str):
    from PIL import Image

    directory = frames_root / scene / "frames"
    colors = sorted((directory / "color").glob("*.jpg"),
                    key=lambda path: int(path.stem))[::25]
    paths, poses = [], []
    for path in colors:
        pose_path = directory / "pose" / f"{path.stem}.txt"
        if not pose_path.exists():
            continue
        pose = np.loadtxt(pose_path).reshape(4, 4)
        if np.isfinite(pose).all():
            paths.append(path)
            poses.append(pose)
    if not paths:
        raise ValueError(f"no valid keyframes: {scene}")
    with Image.open(paths[0]) as image:
        width, height = image.size
    intrinsic = np.loadtxt(
        directory / "intrinsic/intrinsic_color.txt"
    )[:3, :3]
    return paths, poses, intrinsic, width, height


def classify(scenes: list[str], names: list[str], prediction_root: Path,
             frames_root: Path, out: Path, batch_size: int,
             prediction_subdir: str = "full_composite",
             cache_name: str = "semantic_cache") -> None:
    import open_clip
    import torch
    from PIL import Image

    cache = out / cache_name
    cache.mkdir(parents=True, exist_ok=True)
    model = preprocess = text = None
    for scene_index, scene in enumerate(scenes):
        rows = read_rows(prediction_root / prediction_subdir /
                         f"{scene}_boxes.pkl")
        boxes = np.asarray([row[1] for row in rows], np.float64).reshape(-1, 8, 3)
        geometry_hash = hashlib.sha256(boxes.tobytes()).hexdigest()
        destination = cache / f"{scene}.npz"
        if destination.exists():
            with np.load(destination) as old:
                if str(old["geometry_hash"]) != geometry_hash:
                    raise AssertionError(f"semantic cache geometry drift: {scene}")
            print(f"CACHED {scene_index + 1}/{len(scenes)} {scene}", flush=True)
            continue
        start = time.perf_counter()
        if model is None:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-H-14",
                pretrained=str(ROOT / "models/open_clip_pytorch_model.bin"),
            )
            model.cuda().eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                text = model.encode_text(open_clip.tokenize(names).cuda()).float()
                text /= text.norm(dim=-1, keepdim=True)

        paths, poses, intrinsic, width, height = keyframes(frames_root, scene)
        view_indices, rectangles = best_view(
            boxes, poses, intrinsic, width, height
        )
        features = np.zeros((len(boxes), text.shape[1]), np.float32)
        labels = np.full(len(boxes), -1, np.int64)
        similarities = np.full(len(boxes), np.nan, np.float32)
        view_ids = np.full(len(boxes), -1, np.int64)
        valid = np.flatnonzero(view_indices >= 0)
        images = {}
        crop_hashes = []
        gpu_seconds = 0.0
        for begin in range(0, len(valid), batch_size):
            ids = valid[begin:begin + batch_size]
            tensors = []
            for box_index in ids:
                path = paths[view_indices[box_index]]
                if path not in images:
                    with Image.open(path) as image:
                        images[path] = image.convert("RGB")
                x1, y1, x2, y2 = rectangles[box_index]
                dx, dy = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
                crop = images[path].crop((
                    int(max(0, x1 - dx)), int(max(0, y1 - dy)),
                    int(min(width, x2 + dx)), int(min(height, y2 + dy)),
                ))
                tensors.append(preprocess(crop))
                view_ids[box_index] = int(path.stem)
                crop_hashes.append(hashlib.sha256(crop.tobytes()).hexdigest())
            torch.cuda.synchronize()
            tick = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                image_features = model.encode_image(torch.stack(tensors).cuda()).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                cosine = image_features @ text.T
                value, label = cosine.max(dim=-1)
            torch.cuda.synchronize()
            gpu_seconds += time.perf_counter() - tick
            features[ids] = image_features.cpu().numpy()
            labels[ids] = label.cpu().numpy()
            similarities[ids] = value.cpu().numpy()
        np.savez_compressed(
            destination,
            features=features,
            labels=labels,
            max_cosine=similarities,
            view_ids=view_ids,
            rectangles=rectangles,
            geometry_hash=geometry_hash,
            crop_hashes=np.asarray(crop_hashes),
        )
        write_json(cache / f"{scene}.timing.json", {
            "scene": scene,
            "boxes": len(boxes),
            "classified": len(valid),
            "abstained": int((labels < 0).sum()),
            "clip_seconds": gpu_seconds,
            "semantic_stage_seconds": time.perf_counter() - start,
        })
        print(
            f"CLIP {scene_index + 1}/{len(scenes)} {scene}: "
            f"{len(valid)}/{len(boxes)} GPU={gpu_seconds:.2f}s",
            flush=True,
        )


def evaluate(scenes: list[str], split: dict, prediction_root: Path,
             scans_root: Path, out: Path) -> dict:
    names = split["all_names"]
    name_to_index = {name: index for index, name in enumerate(names)}
    group_ids = {
        "base": [name_to_index[name] for name in split["base_names"]],
        "novel": [name_to_index[name] for name in split["novel_names"]],
        "all": list(range(len(names))),
    }
    gt_boxes, gt_classes = {}, {}
    predictions = {"BoxFusion-Ctrl": {}, "ReCaR-3D": {}}
    pred_classes = {"BoxFusion-Ctrl": {}, "ReCaR-3D": {}}
    abstained = Counter()
    prefix_audit = {}
    for scene in scenes:
        with np.load(out / "gt_cache" / f"{scene}.npz") as gt:
            gt_boxes[scene] = gt["boxes"].copy()
            gt_classes[scene] = gt["classes"].copy()
        native = read_rows(prediction_root / "native" / f"{scene}_boxes.pkl")
        full = read_rows(prediction_root / "full_composite" /
                         f"{scene}_boxes.pkl")
        if len(full) < len(native):
            raise AssertionError(f"full shorter than native: {scene}")
        for first, second in zip(native, full[:len(native)]):
            if not np.array_equal(first[1], second[1]):
                raise AssertionError(f"native is not full geometry prefix: {scene}")
        with np.load(out / "semantic_cache" / f"{scene}.npz") as semantic:
            labels = semantic["labels"].copy()
        transform = alignment(scans_root, scene)
        for arm, rows, arm_labels in (
            ("BoxFusion-Ctrl", native, labels[:len(native)]),
            ("ReCaR-3D", full, labels),
        ):
            boxes = np.asarray([row[1] for row in rows], np.float64).reshape(-1, 8, 3)
            boxes = boxes @ transform[:3, :3].T + transform[:3, 3]
            scores = np.asarray([float(row[2]) for row in rows], np.float64)
            predictions[arm][scene] = (boxes, scores)
            pred_classes[arm][scene] = arm_labels
            abstained[arm] += int((arm_labels < 0).sum())
        prefix_audit[scene] = {"native": len(native), "full": len(full)}

    result = {
        "schema": "boxfusion.scannet100.frequency_group.metrics.v2",
        "scene_count": len(scenes),
        "scene_scope": "paper ScanNet-100 subset, not full CoDA validation",
        "claim_boundary": (
            "The system was not trained with CoDA base-only supervision. Base "
            "and novel keys are retained for compatibility but mean "
            "frequent-10 and tail-50, not seen and unseen."
        ),
        "semantic_protocol": (
            "largest valid projected gap-25 keyframe crop with 15% padding; "
            "frozen CLIP ViT-H-14; argmax normalized cosine over the joint "
            "60-class vocabulary; unchanged 3D detection score for AP ranking"
        ),
        "groups": {key: [names[index] for index in value]
                   for key, value in group_ids.items()},
        "arms": {},
        "prefix_audit": prefix_audit,
    }
    for arm in predictions:
        per_class = {}
        for class_id, name in enumerate(names):
            class_gt = {scene: gt_boxes[scene][gt_classes[scene] == class_id]
                        for scene in scenes}
            class_predictions = {
                scene: (boxes[pred_classes[arm][scene] == class_id],
                        scores[pred_classes[arm][scene] == class_id])
                for scene, (boxes, scores) in predictions[arm].items()
            }
            gt_count = sum(len(value) for value in class_gt.values())
            row = {
                "split": "base" if class_id in group_ids["base"] else "novel",
                "gt": gt_count,
                "predictions": sum(len(value[0])
                                   for value in class_predictions.values()),
            }
            for threshold in THRESHOLDS:
                row[str(threshold)] = (
                    class_agnostic_ap(class_predictions, class_gt, threshold)
                    if gt_count else None
                )
            per_class[name] = row

        aggregates = {}
        for group, ids in group_ids.items():
            active = [per_class[names[index]] for index in ids
                      if per_class[names[index]]["gt"] > 0]
            aggregates[group] = {
                "defined_classes": len(ids),
                "active_classes": len(active),
                "gt": sum(row["gt"] for row in active),
                "AP": {
                    str(threshold): float(np.mean([
                        row[str(threshold)]["ap"] for row in active
                    ])) if active else None
                    for threshold in THRESHOLDS
                },
            }
        base25 = aggregates["base"]["AP"]["0.25"]
        novel25 = aggregates["novel"]["AP"]["0.25"]
        harmonic25 = (2 * base25 * novel25 / (base25 + novel25)
                      if base25 + novel25 else 0.0)
        result["arms"][arm] = {
            "boxes": sum(len(value[0]) for value in predictions[arm].values()),
            "abstained": int(abstained[arm]),
            "aggregates": aggregates,
            "harmonic_AP25_base_novel": harmonic25,
            "per_class": per_class,
        }
    write_json(out / "metrics.json", result)
    return result


def render_report(result: dict, split: dict, out: Path) -> None:
    lines = [
        "# ScanNet-100类别频次分组诊断（不构成base/novel开放词汇评测）",
        "",
        "> **主张边界：** 当前冻结系统没有采用CoDA的base-only监督训练，",
        "> 因此下文兼容字段中的base/novel实际只表示高频10类/其余50类，",
        "> 不能解释为训练时见过/未见过类别，也不能作为开放词汇泛化主结果。",
        "",
        "评测场景是本文固定的ScanNet-100子集，并非CoDA完整ScanNet验证集。",
        "",
        "## 类别划分",
        "",
        "- Frequent-10（兼容键base）: " + ", ".join(split["base_names"]),
        "- Tail-50（兼容键novel）: " + ", ".join(split["novel_names"]),
        "",
        "## 结果",
        "",
        "| Method | APfreq@.15 | APfreq@.25 | APfreq@.50 | APtail@.15 | APtail@.25 | APtail@.50 | APall@.25 | H@.25 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ("BoxFusion-Ctrl", "ReCaR-3D"):
        value = result["arms"][arm]
        agg = value["aggregates"]
        lines.append(
            f"| {arm} | {agg['base']['AP']['0.15']:.4f} | "
            f"{agg['base']['AP']['0.25']:.4f} | {agg['base']['AP']['0.5']:.4f} | "
            f"{agg['novel']['AP']['0.15']:.4f} | {agg['novel']['AP']['0.25']:.4f} | "
            f"{agg['novel']['AP']['0.5']:.4f} | {agg['all']['AP']['0.25']:.4f} | "
            f"{value['harmonic_AP25_base_novel']:.4f} |"
        )
    base = result["arms"]["BoxFusion-Ctrl"]["aggregates"]
    full = result["arms"]["ReCaR-3D"]["aggregates"]
    delta = {
        group: {threshold: full[group]["AP"][threshold] - base[group]["AP"][threshold]
                for threshold in ("0.15", "0.25", "0.5")}
        for group in ("base", "novel", "all")
    }
    lines += [
        "",
        ("ReCaR-3D相对BoxFusion-Ctrl的AP25增量为："
         f"frequent-10 **{delta['base']['0.25']:+.4f}**，"
         f"tail-50 **{delta['novel']['0.25']:+.4f}**，"
         f"all **{delta['all']['0.25']:+.4f}**。"),
        "",
        "宏平均只包含该100场子集中实际出现GT的类别；类别数记录在metrics.json。",
        "语义读取采用最大有效投影关键帧、15%裁剪边距、冻结CLIP ViT-H-14，",
        "在base与novel联合60类词表上取余弦相似度argmax，AP排序沿用原三维检测分数。",
        "ReCaR-3D未为此评测训练或调节任何参数。",
        "",
        "## 复现",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 python tools/run_scannet_base_novel_eval.py",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")

    with (out / "per_class.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "class", "split", "gt",
            "ctrl_predictions", "ctrl_ap15", "ctrl_ap25", "ctrl_ap50",
            "recar_predictions", "recar_ap15", "recar_ap25", "recar_ap50",
            "delta_ap25",
        ])
        ctrl_rows = result["arms"]["BoxFusion-Ctrl"]["per_class"]
        recar_rows = result["arms"]["ReCaR-3D"]["per_class"]
        for name in split["all_names"]:
            ctrl, recar = ctrl_rows[name], recar_rows[name]
            writer.writerow([
                name, ctrl["split"], ctrl["gt"], ctrl["predictions"],
                ctrl["0.15"]["ap"], ctrl["0.25"]["ap"], ctrl["0.5"]["ap"],
                recar["predictions"], recar["0.15"]["ap"],
                recar["0.25"]["ap"], recar["0.5"]["ap"],
                recar["0.25"]["ap"] - ctrl["0.25"]["ap"],
            ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PRED_ROOT)
    parser.add_argument("--scans-root", type=Path, default=DEFAULT_SCANS)
    parser.add_argument("--meta-root", type=Path, default=DEFAULT_META)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    official = (ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt").read_text().split()
    available = sorted(path.name.replace("_boxes.pkl", "") for path in
                       (args.prediction_root / "native").glob("*_boxes.pkl"))
    if len(available) != 100 or set(available) != set(official):
        raise AssertionError("prediction set is not the frozen official ScanNet-100 set")
    scenes = available[:args.limit]
    split = build_split(args.meta_root, args.scans_root)
    write_json(args.output / "split.json", split)

    hashes = {}
    for arm in ("native", "full_composite"):
        for scene in scenes:
            path = args.prediction_root / arm / f"{scene}_boxes.pkl"
            hashes[str(path.resolve())] = sha256(path)
    hashes[str((ROOT / "models/open_clip_pytorch_model.bin").resolve())] = sha256(
        ROOT / "models/open_clip_pytorch_model.bin"
    )
    write_json(args.output / "source_hashes.json", hashes)
    write_json(args.output / "source_manifest.json", {
        "prediction_root": str(args.prediction_root.resolve()),
        "scans_root": str(args.scans_root.resolve()),
        "frames_root": str(args.frames_root.resolve()),
        "scenes": scenes,
        "scene_count": len(scenes),
        "git_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "CoDA-style class split evaluated on frozen paper ScanNet-100",
    })

    if not all((args.output / "gt_cache" / f"{scene}.npz").exists()
               for scene in scenes):
        prepare_gt(scenes, split, args.scans_root, args.meta_root, args.output)
    if args.prepare_only:
        return
    if not args.evaluate_only:
        classify(scenes, split["all_names"], args.prediction_root,
                 args.frames_root, args.output, args.batch_size)
    result = evaluate(scenes, split, args.prediction_root,
                      args.scans_root, args.output)
    render_report(result, split, args.output)
    for arm, value in result["arms"].items():
        print(arm, value["aggregates"], flush=True)


if __name__ == "__main__":
    main()
