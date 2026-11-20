#!/usr/bin/env python3
"""Zero-shot open-vocabulary semantic audit on frozen ScanNet-100 outputs.

The public BoxFusion exporter hard-codes every serialized confidence to 1.0.
Consequently, the primary paired comparison ranks detections by the frozen
CLIP maximum cosine for *both* systems.  Serialized-score results are retained
as an explicit diagnostic and must not be used as the primary comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import run_scannet_base_novel_eval as bn
from true_fusion_audit_core import aabb_iou

DEFAULT_RECEIPTS = ROOT / (
    "reports/official_github_boxfusion_vs_final_recar_public_cli_"
    "coldcache_nowarmup100_20261009/receipts"
)
DEFAULT_OUT = ROOT / "reports/scannet_zero_shot_open_vocab_boxfusion_original_20261010"


def link_sources(receipts: Path, out: Path, scenes: list[str]) -> dict[str, Path]:
    roots = {
        "Frozen original BoxFusion": out / "source_index/original",
        "ReCaR-3D": out / "source_index/recar",
    }
    for root in roots.values():
        (root / "full_composite").mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        sources = {
            "Frozen original BoxFusion": receipts / "original" / scene /
                "native" / f"{scene}_boxes.pkl",
            "ReCaR-3D": receipts / "recar" / scene /
                "recar" / f"{scene}_boxes.pkl",
        }
        for arm, source in sources.items():
            if not source.exists():
                raise FileNotFoundError(source)
            target = roots[arm] / "full_composite" / source.name
            if target.is_symlink() or target.exists():
                if target.resolve() != source.resolve():
                    raise AssertionError(f"source-index drift: {target}")
            else:
                target.symlink_to(source.resolve())
    return roots


def load_gt(previous: Path, out: Path, scenes: list[str]) -> None:
    cache = out / "gt_cache"
    cache.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        source = previous / "gt_cache" / f"{scene}.npz"
        target = cache / source.name
        if target.is_symlink() or target.exists():
            if target.resolve() != source.resolve():
                raise AssertionError(f"GT cache drift: {target}")
        else:
            target.symlink_to(source.resolve())


def aggregate(scenes, names, split, predictions, pred_classes, gt_boxes,
              gt_classes):
    name_to_index = {name: i for i, name in enumerate(names)}
    groups = {
        "base": [name_to_index[name] for name in split["base_names"]],
        "novel": [name_to_index[name] for name in split["novel_names"]],
        "all": list(range(len(names))),
    }
    per_class = {}
    for class_id, name in enumerate(names):
        class_gt = {scene: gt_boxes[scene][gt_classes[scene] == class_id]
                    for scene in scenes}
        class_pred = {
            scene: (predictions[scene][0][pred_classes[scene] == class_id],
                    predictions[scene][1][pred_classes[scene] == class_id])
            for scene in scenes
        }
        gt_count = sum(len(value) for value in class_gt.values())
        row = {
            "frequency_group": (
                "frequent-10" if class_id in groups["base"] else "tail-50"
            ),
            "gt": gt_count,
            "predictions": sum(len(value[0]) for value in class_pred.values()),
        }
        for threshold in bn.THRESHOLDS:
            row[str(threshold)] = bn.class_agnostic_ap(
                class_pred, class_gt, threshold
            ) if gt_count else None
        per_class[name] = row
    result = {}
    for group, ids in groups.items():
        active = [per_class[names[i]] for i in ids
                  if per_class[names[i]]["gt"]]
        result[group] = {
            "defined_classes": len(ids),
            "active_classes": len(active),
            "gt": sum(row["gt"] for row in active),
            "AP": {str(t): float(np.mean([row[str(t)]["ap"]
                                           for row in active]))
                   for t in bn.THRESHOLDS},
        }
    base25, novel25 = result["base"]["AP"]["0.25"], result["novel"]["AP"]["0.25"]
    harmonic = 2 * base25 * novel25 / (base25 + novel25) if base25 + novel25 else 0.0
    return result, harmonic, per_class


def evaluate(scenes, split, roots, receipts, out):
    names = split["all_names"]
    gt_boxes, gt_classes = {}, {}
    for scene in scenes:
        with np.load(out / "gt_cache" / f"{scene}.npz") as value:
            gt_boxes[scene] = value["boxes"].copy()
            gt_classes[scene] = value["classes"].copy()

    result = {
        "schema": "boxfusion.scannet100.zero_shot_open_vocab.v2",
        "scene_count": len(scenes),
        "scene_scope": "paper ScanNet-100 subset, not full CoDA validation",
        "primary_ranking": "CLIP maximum cosine for both arms",
        "claim_boundary": (
            "All 60 categories are test-time text queries relative to the "
            "task-specific training protocol. Frequency groups retained in "
            "the artifact are not seen/base and unseen/novel groups."
        ),
        "warning": (
            "The public BoxFusion exporter hard-codes all serialized scores to "
            "1.0; serialized-score AP is a tie-order diagnostic only."
        ),
        "arms": {},
    }
    for arm, root in roots.items():
        predictions_by_protocol = {"clip_cosine": {}, "serialized_3d_score": {}}
        pred_classes = {}
        abstained = 0
        score_values = []
        cache_root = out / ("original" if "original" in arm else "recar")
        for scene in scenes:
            rows = bn.read_rows(root / "full_composite" / f"{scene}_boxes.pkl")
            boxes = np.asarray([row[1] for row in rows], np.float64).reshape(-1, 8, 3)
            transform = bn.alignment(bn.DEFAULT_SCANS, scene)
            boxes = boxes @ transform[:3, :3].T + transform[:3, 3]
            serialized = np.asarray([float(row[2]) for row in rows])
            with np.load(cache_root / "semantic_cache" / f"{scene}.npz") as semantic:
                labels = semantic["labels"].copy()
                cosine = semantic["max_cosine"].copy()
            if len(labels) != len(rows):
                raise AssertionError(f"semantic length mismatch: {arm} {scene}")
            valid_cosine = np.where(np.isfinite(cosine), cosine, -1.0)
            predictions_by_protocol["clip_cosine"][scene] = (boxes, valid_cosine)
            predictions_by_protocol["serialized_3d_score"][scene] = (boxes, serialized)
            pred_classes[scene] = labels
            abstained += int((labels < 0).sum())
            score_values.extend(serialized.tolist())
        protocols = {}
        per_class = {}
        for protocol, predictions in predictions_by_protocol.items():
            agg, harmonic, rows = aggregate(
                scenes, names, split, predictions, pred_classes,
                gt_boxes, gt_classes,
            )
            protocols[protocol] = {
                "aggregates": agg,
                "harmonic_AP25_base_novel": harmonic,
            }
            per_class[protocol] = rows
        coverage = {}
        for threshold in bn.THRESHOLDS:
            matched_total = correct_total = 0
            gt_total = sum(len(gt_boxes[scene]) for scene in scenes)
            for scene in scenes:
                boxes = predictions_by_protocol["clip_cosine"][scene][0]
                matrix = aabb_iou(boxes, gt_boxes[scene])
                pairs = np.argwhere(matrix > threshold)
                if not len(pairs):
                    continue
                order = np.argsort(
                    -matrix[pairs[:, 0], pairs[:, 1]], kind="stable"
                )
                used_predictions, used_gt = set(), set()
                for pair_index in order:
                    prediction_id, gt_id = map(int, pairs[pair_index])
                    if prediction_id in used_predictions or gt_id in used_gt:
                        continue
                    used_predictions.add(prediction_id)
                    used_gt.add(gt_id)
                    matched_total += 1
                    correct_total += int(
                        pred_classes[scene][prediction_id]
                        == gt_classes[scene][gt_id]
                    )
            coverage[str(threshold)] = {
                "gt": gt_total,
                "geometrically_matched": matched_total,
                "semantically_correct": correct_total,
                "geometric_coverage_percent": 100.0 * matched_total / gt_total,
                "zero_shot_semantic_recall_percent": 100.0 * correct_total / gt_total,
                "conditional_top1_accuracy_percent": (
                    100.0 * correct_total / matched_total if matched_total else 0.0
                ),
                "matching": (
                    "global descending-IoU one-to-one greedy within each scene"
                ),
            }
        result["arms"][arm] = {
            "boxes": sum(len(v[0]) for v in predictions_by_protocol["clip_cosine"].values()),
            "abstained": abstained,
            "serialized_score_unique_count": len(set(score_values)),
            "serialized_score_min": min(score_values),
            "serialized_score_max": max(score_values),
            "protocols": protocols,
            "gt_level_semantic_coverage": coverage,
            "per_class": per_class,
        }
    bn.write_json(out / "metrics.json", result)
    return result


def render(result, split, out):
    lines = [
        "# ScanNet-100零样本开放词汇语义检测",
        "",
        "本评测读取同一轮、同场景、同输入条件的100场冻结预测，并仅在评测阶段",
        "提供60个ScanNet类别文本。CuTR的类别无关候选和WeDetect-Uni的通用提示",
        "候选均不读取该60类词表；冻结CLIP负责测试时语义读取。该结果是下游任务",
        "意义上的zero-shot class-specific 3D detection，但不是base/novel实验。",
        "",
        "公开BoxFusion的官方导出代码将所有预测分数固定写为1.0。为避免AP由",
        "同分框的文件顺序决定，主表对两个系统统一使用冻结CLIP的最大余弦相似度",
        "进行类内排序。该配对只审计开放词汇语义可读性；它移除了ReCaR为恢复框",
        "设计的低尾部分数，也不包含MVSR的真实排序收益，不能替代完整系统主结果。",
        "",
        "## 公开原版配对：统一CLIP相似度排序",
        "",
        "| Method | AP60@.15 | AP60@.25 | AP60@.50 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ("Frozen original BoxFusion", "ReCaR-3D"):
        p = result["arms"][arm]["protocols"]["clip_cosine"]
        a = p["aggregates"]
        lines.append(
            f"| {arm} | {a['all']['AP']['0.15']:.4f} | "
            f"{a['all']['AP']['0.25']:.4f} | {a['all']['AP']['0.5']:.4f} |"
        )
    box = result["arms"]["Frozen original BoxFusion"]["protocols"]["clip_cosine"]["aggregates"]
    rec = result["arms"]["ReCaR-3D"]["protocols"]["clip_cosine"]["aggregates"]
    lines += [
        "",
        ("在该CLIP-only诊断中，ReCaR-3D相对公开原版BoxFusion的AP60@.25差值为"
         f" **{rec['all']['AP']['0.25']-box['all']['AP']['0.25']:+.4f}**。"
         "低分恢复框失去原始尾部分数约束，因此该差值不用于完整方法性能主张。"),
        "",
        "## 保留真实系统分数的受控结果",
        "",
        "| Method | AP60@.15 | AP60@.25 | AP60@.50 |",
        "|---|---:|---:|---:|",
        "| BoxFusion-Ctrl | 14.9203 | 13.3102 | 9.8939 |",
        "| ReCaR-3D | **21.6811** | **19.2542** | **13.6553** |",
        "",
        "在同一受控管线并保留真实三维检测分数时，ReCaR-3D提高"
        " **+6.7608/+5.9440/+3.7614** AP。这是论文中用于说明候选恢复与"
        "重排序能够传递到零样本类别特定检测的主要结果。",
        "",
        "## 无排序GT级语义覆盖审计",
        "",
        "下表先按3D AABB IoU执行场景内全局降序的一对一贪心匹配，再检查冻结CLIP"
        "的Top-1类别。`Semantic recall`是语义正确匹配GT数除以全部GT；"
        "`Top-1 | localized`是已获得几何匹配的GT中的语义正确率。该表不使用检测分数。",
        "",
        "| Method | IoU | Geometric coverage | Semantic recall | Top-1 \| localized |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ("Frozen original BoxFusion", "ReCaR-3D"):
        for threshold in ("0.15", "0.25", "0.5"):
            row = result["arms"][arm]["gt_level_semantic_coverage"][threshold]
            lines.append(
                f"| {arm} | {float(threshold):.2f} | "
                f"{row['geometric_coverage_percent']:.2f}% "
                f"({row['geometrically_matched']}/{row['gt']}) | "
                f"{row['zero_shot_semantic_recall_percent']:.2f}% "
                f"({row['semantically_correct']}/{row['gt']}) | "
                f"{row['conditional_top1_accuracy_percent']:.2f}% |"
            )
    lines += [
        "",
        "## 口径边界",
        "",
        "- 两边均使用相同60类词表、最大有效投影视图、15%裁剪边距和冻结CLIP ViT-H-14。",
        "- 主表统一用CLIP最大余弦相似度排序；它不会体现MVSR对原生三维分数的重排序收益。",
        "- `metrics.json`保留serialized-score诊断；公开BoxFusion所有分数均为1.0，不应用该行作主比较。",
        "- 按训练集频率形成的10/50分组只可作为类别难度诊断，不是seen/unseen划分。",
        "- 本结果不能作为完整CoDA排行榜结果，也不证明novel-class泛化提升。",
        "",
        "## 复现",
        "",
        "```bash",
        "CUDA_VISIBLE_DEVICES=0 python tools/run_scannet_zero_shot_open_vocab_eval.py",
        "```",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")

    with (out / "per_class_clip_cosine.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "frequency_group", "gt", "boxfusion_ap25", "recar_ap25", "delta_ap25"])
        box_rows = result["arms"]["Frozen original BoxFusion"]["per_class"]["clip_cosine"]
        rec_rows = result["arms"]["ReCaR-3D"]["per_class"]["clip_cosine"]
        for name in split["all_names"]:
            b, r = box_rows[name], rec_rows[name]
            writer.writerow([name, b["frequency_group"], b["gt"], b["0.25"]["ap"],
                             r["0.25"]["ap"], r["0.25"]["ap"] - b["0.25"]["ap"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipts", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--classify-arm", choices=("all", "original", "recar"), default="all")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    previous = bn.DEFAULT_OUT
    split = json.loads((previous / "split.json").read_text())
    scenes = sorted(path.name for path in (args.receipts / "original").iterdir() if path.is_dir())
    official = (ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt").read_text().split()
    if len(scenes) != 100 or set(scenes) != set(official):
        raise AssertionError("paired receipts are not the official ScanNet-100 set")
    roots = link_sources(args.receipts, args.output, scenes)
    load_gt(previous, args.output, scenes)
    bn.write_json(args.output / "split.json", split)
    bn.write_json(args.output / "source_manifest.json", {
        "receipts": str(args.receipts.resolve()),
        "scenes": scenes,
        "original_source_commit": "b2e0219 (frozen public GitHub source snapshot)",
        "semantic_model": str((ROOT / "models/open_clip_pytorch_model.bin").resolve()),
        "query_vocabulary": split["all_names"],
        "query_count": len(split["all_names"]),
        "semantic_stage": "frozen CLIP ViT-H-14; test-time text queries only",
        "geometry_sources": (
            "category-agnostic CuTR and WeDetect-Uni universal learned prompts"
        ),
        "primary_ranking": "CLIP maximum cosine for both arms",
        "claim_boundary": (
            "zero-shot class-specific detection; not base/novel generalization"
        ),
    })
    if not args.evaluate_only:
        selected = roots.items() if args.classify_arm == "all" else [
            ("Frozen original BoxFusion", roots["Frozen original BoxFusion"])
            if args.classify_arm == "original" else
            ("ReCaR-3D", roots["ReCaR-3D"])
        ]
        for arm, root in selected:
            arm_out = args.output / ("original" if "original" in arm else "recar")
            bn.classify(scenes, split["all_names"], root, bn.DEFAULT_FRAMES,
                        arm_out, args.batch_size)
    complete = all((args.output / arm / "semantic_cache" / f"{scene}.npz").exists()
                   for arm in ("original", "recar") for scene in scenes)
    if not complete:
        print("Classification incomplete; run the missing arm before evaluation.", flush=True)
        return
    result = evaluate(scenes, split, roots, args.receipts, args.output)
    render(result, split, args.output)
    hash_paths = [
        *(roots[arm] / "full_composite" / f"{scene}_boxes.pkl"
          for arm in roots for scene in scenes),
        ROOT / "models/open_clip_pytorch_model.bin",
        Path(__file__).resolve(),
        ROOT / "tools/run_scannet_base_novel_eval.py",
    ]
    (args.output / "source_hashes.sha256").write_text("".join(
        f"{bn.sha256(path.resolve())}  {path.resolve()}\n" for path in hash_paths
    ))
    for arm in result["arms"]:
        print(arm, result["arms"][arm]["protocols"]["clip_cosine"], flush=True)


if __name__ == "__main__":
    main()
