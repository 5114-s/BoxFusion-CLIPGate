#!/usr/bin/env python3
"""Frozen-cache ScanNet source-of-miss audit and causal memory budget controls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_scannet_fixed_capacity_prefix_memory as core
from tools.bootstrap_final_recar_pairs import scene_pairs, ap_from_pairs
from tools.true_fusion_audit_core import aabb_iou


def read_rows(path):
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)[0]
    return [(int(label), np.asarray(box, dtype=np.float64), float(score))
            for label, box, score in data]


def rows_from_snapshot(snapshot):
    return [(0, np.array(box, copy=True), float(score))
            for box, score in zip(snapshot.boxes, snapshot.scores)]


def maximum_overlap(boxes, gt):
    if not len(boxes):
        return np.zeros(len(gt), dtype=np.float64)
    return aabb_iou(boxes, gt).max(axis=0)


def lengths(rows):
    counts = Counter(rows.sources)
    return {"native": counts["native"], "proposal": counts["m1p"],
            "anchor": counts["m1a"], "total": len(rows.boxes)}


def state_counts(full, proposal, anchor):
    return {
        "full_plr_active": len(full.m1p.tracks),
        "full_plr_archive": len(full.m1p.births),
        "full_calr_active": len(full.m1a_control.active),
        "full_calr_births": len(full.m1a_control.births),
        "hcsr_support_states": len(full.m2.reliability.states),
        "hcsr_geometry_states": len(full.m2.geometry),
        "memory_proposal_active": len(proposal.active),
        "memory_proposal_archive": len(proposal.archive),
        "memory_anchor": len(anchor.state),
    }


def update_peaks(peaks, values):
    for name, value in values.items():
        peaks[name] = max(peaks.get(name, 0), value)
    peaks["full_recovery_total"] = max(
        peaks.get("full_recovery_total", 0),
        sum(values[k] for k in ("full_plr_active", "full_plr_archive",
                               "full_calr_active", "full_calr_births")),
    )
    peaks["memory_recovery_total"] = max(
        peaks.get("memory_recovery_total", 0),
        sum(values[k] for k in ("memory_proposal_active", "memory_proposal_archive",
                               "memory_anchor")),
    )


def compare_runs(scenes, current_root):
    roots = {
        "main_old": ROOT / "development/final_controls/experiments/mvsr_design_20261005/factorial_no_visibility",
        "canonical": ROOT / "development/final_controls/experiments/scannet_factorial_1787_canonical_20261012",
        "prefix_replay": current_root,
    }
    by_run = {key: {} for key in roots}
    counts = {}
    hashes = {}
    for name, root in roots.items():
        total = {"native": 0, "full": 0}
        for scene in scenes:
            base_path = root / ("base_p100" if name == "prefix_replay" else "base") / f"{scene}_boxes.pkl"
            full_path = root / ("full_p100" if name == "prefix_replay" else "full") / f"{scene}_boxes.pkl"
            native, full = read_rows(base_path), read_rows(full_path)
            by_run[name][scene] = (native, full)
            total["native"] += len(native)
            total["full"] += len(full)
            for path in (base_path, full_path):
                hashes[str(path)] = core.sha256(path)
        counts[name] = total
    comparisons = {}
    for left, right in (("main_old", "canonical"), ("canonical", "prefix_replay")):
        row = {"native_count_different_scenes": 0, "native_geometry_exact_scenes": 0,
               "native_scores_exact_scenes": 0, "native_max_corner_error_where_counts_match": 0.,
               "full_exact_scenes": 0, "full_count_different_scenes": 0}
        for scene in scenes:
            a, af = by_run[left][scene]
            b, bf = by_run[right][scene]
            if len(a) != len(b):
                row["native_count_different_scenes"] += 1
            else:
                geometric = all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))
                row["native_geometry_exact_scenes"] += int(geometric)
                row["native_scores_exact_scenes"] += int(all(x[2] == y[2] for x, y in zip(a, b)))
                if a:
                    row["native_max_corner_error_where_counts_match"] = max(
                        row["native_max_corner_error_where_counts_match"],
                        max(float(np.max(np.abs(x[1]-y[1]))) for x, y in zip(a, b)),
                    )
            row["full_count_different_scenes"] += int(len(af) != len(bf))
            row["full_exact_scenes"] += int(
                len(af) == len(bf) and all(
                    x[0] == y[0] and x[2] == y[2] and np.array_equal(x[1], y[1])
                    for x, y in zip(af, bf)
                )
            )
        comparisons[f"{left}_vs_{right}"] = row
    return {
        "roots": {k: str(v) for k, v in roots.items()}, "counts": counts,
        "comparisons": comparisons, "prediction_sha256": hashes,
        "causal_attribution_limit": (
            "Differences in frozen native records and full output files are verified. "
            "The files alone do not identify whether a particular upstream numerical "
            "difference was caused by random sampling, CUDA nondeterminism, or scheduling."
        ),
    }


def bootstrap(predictions, gts, scenes, comparisons, replicates, seed):
    thresholds = core.THRESHOLDS
    needed = sorted({name for pair in comparisons.values() for name in pair})
    pairs = {
        arm: {t: {scene: scene_pairs(*predictions[arm][scene], gts[scene], t)
                  for scene in scenes} for t in thresholds}
        for arm in needed
    }
    gt_sizes = np.asarray([len(gts[s]) for s in scenes])
    draws = np.random.default_rng(seed).integers(0, len(scenes), (replicates, len(scenes)))
    samples = {name: {str(t): [] for t in thresholds} for name in comparisons}
    for index, draw in enumerate(draws):
        selected = [scenes[int(i)] for i in draw]
        gt_total = int(gt_sizes[draw].sum())
        values = {
            arm: {t: ap_from_pairs(np.concatenate([pairs[arm][t][s] for s in selected]), gt_total)
                  for t in thresholds}
            for arm in needed
        }
        for name, (treatment, base) in comparisons.items():
            for t in thresholds:
                samples[name][str(t)].append(values[treatment][t]-values[base][t])
        if (index+1) % 250 == 0:
            print(f"bootstrap {index+1}/{replicates}", flush=True)
    return {
        name: {t: {"ci95_ap": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                   "p_positive": float(np.mean(np.asarray(a) > 0))}
               for t, a in by_threshold.items()}
        for name, by_threshold in samples.items()
    }


def make_report(result):
    lines = [
        "# ScanNet-100：恢复来源、原生尺寸过滤与简单因果记忆预算", "",
        "本实验统一使用同一份冻结逐关键帧原生地图缓存和辅助候选缓存，CPU顺序重放；"
        "未重新运行CuTR、Boxer或辅助检测器。全部AP仅在本次受控重放内比较。", "",
        "## 1. 恢复来源审计", "",
        "按每个GT的最大3D AABB IoU统计覆盖，严格使用IoU>阈值。"
        "这不是分数排序、一对一匹配下的标准检测召回率。"
        "PLR/CALR列可重叠；任一分支为并集。恢复集合采用0.30 m原生读出工作点的实际终态分支输出。", "",
        "| IoU | 原生输出遗漏来源 | GT数 | PLR覆盖 | CALR覆盖 | 任一分支覆盖 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for threshold, categories in result["recovery_origin"].items():
        for category, title in (("internal_native_miss", "过滤前原生地图也未覆盖"),
                                ("removed_by_readout_filter", "过滤前已覆盖、过滤后未覆盖")):
            row = categories[category]
            lines.append(f"| {threshold} | {title} | {row['gt']} | {row['plr']} | {row['calr']} | {row['either']} |")
    lines += ["", "## 2. 放宽原生读出尺寸过滤", "",
              "只改变原生输出尺寸规则，并针对对应原生输出重算PLR覆盖排除和读出；"
              "内部原生状态、HCSR历史、候选证据及CALR均保持相同。0表示关闭尺寸过滤。", "",
              "| 逐轴最小尺寸(m) | 方法 | 原生/PLR/CALR输出 | 总输出 | AP15 | AP25 | AP50 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for tag, threshold in (("f030", .30), ("f015", .15), ("f000", .0)):
        for system in ("base", "full"):
            arm = f"{system}_{tag}"
            row, ap = result["terminal_counts"][arm], result["terminal_ap"][arm]
            lines.append(f"| {threshold:.2f} | {system} | {row['native']:,}/{row['proposal']:,}/{row['anchor']:,} | {row['total']:,} | {ap['0.15']:.4f} | {ap['0.25']:.4f} | {ap['0.5']:.4f} |")
    lines += ["", "## 3. 预先固定的简单记忆读出预算", "",
              "简单基线为分支分离的最高分体素记忆：每帧读取同一缓存中的最多150个post-NMS proposals和300个pre-NMS anchors，"
              "以0.30 m体素聚合中心，每个体素保留原始分数最高的单次观测，不执行三关键帧确认或medoid选择。"
              "proposal读出保留0.25原生覆盖排除、0.50分支内NMS和相同尺寸分桶出生分数；anchor采用相同低分尾部映射。"
              "proposal活动状态的失效期限为10个关键帧，归档则受独立容量约束。", "",
              "proposal上限统一为12；anchor上限128/256/512于新增结果产生前固定，640为已存在工作点的复核。"
              "不读取各场景Full终态数量来配额。固定状态容量为proposal活动1024、归档1024、anchor4096。"
              "所有预算逐关键帧读出，共享相同三因素HCSR。", "",
              "这里的预算是每个场景在每个已提交关键帧的当前读出上限，不是累计释放上限；"
              "输出数列统计100场景终态记录总数，并包含共享的1,787条原生记录。"
              "预算围绕此前已知输出规模预设，属于同一测试集上的事后敏感性分析，不是新的盲测。", "",
              "| 恢复策略 | 预设P/A上限 | 实际P/A输出 | 总输出 | AP15 | AP25 | AP50 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for cap in result["protocol"]["memory_anchor_output_caps"]:
        arm = f"memory_a{cap}"
        row, ap = result["terminal_counts"][arm], result["terminal_ap"][arm]
        lines.append(f"| 简单因果记忆 | 12/{cap} | {row['proposal']:,}/{row['anchor']:,} | {row['total']:,} | {ap['0.15']:.4f} | {ap['0.25']:.4f} | {ap['0.5']:.4f} |")
    row, ap = result["terminal_counts"]["full_f030"], result["terminal_ap"]["full_f030"]
    lines.append(f"| 完整恢复策略 | 12/640（确认后） | {row['proposal']:,}/{row['anchor']:,} | {row['total']:,} | {ap['0.15']:.4f} | {ap['0.25']:.4f} | {ap['0.5']:.4f} |")
    lines += ["", "## 4. 逐前缀AP", "",
              "前缀按已处理关键帧数量的25%、50%、75%、100%取整读取当前已提交状态，"
              "所有前缀使用相同的完整场景GT集合（共1,433个），未按当前可见性裁剪GT。"
              "因此该表衡量全场景目标的阶段覆盖，不等同于首次可见至检测的延迟或可见目标AP。"
              "100%前缀与本次重放终态预测一致。", "",
              "| 前缀 | 策略 | 实际P/A输出 | 总输出 | AP15 | AP25 | AP50 |",
              "|---:|---|---:|---:|---:|---:|---:|"]
    for fraction, arms in result["prefix_ap"].items():
        for arm, ap in arms.items():
            row = result["prefix_counts"][fraction][arm]
            lines.append(f"| {int(float(fraction)*100)}% | {arm} | {row['proposal']:,}/{row['anchor']:,} | {row['total']:,} | {ap['0.15']:.4f} | {ap['0.25']:.4f} | {ap['0.5']:.4f} |")
    lines += ["", "## 5. 实际状态条目峰值", "",
              "| 状态 | 完整方法 | 简单记忆（所有读出预算相同） |",
              "|---|---:|---:|"]
    p = result["state_peaks"]
    for title, left, right in (("Proposal活动", "full_plr_active", "memory_proposal_active"),
                                ("Proposal归档", "full_plr_archive", "memory_proposal_archive"),
                                ("Anchor未确认/记忆体素", "full_calr_active", "memory_anchor")):
        lines.append(f"| {title} | {p[left]} | {p[right]} |")
    lines += [f"| Anchor确认记录 | {p['full_calr_births']} | 包含于anchor记忆体素 |",
              f"| 恢复状态同时占用总峰值 | {p['full_recovery_total']} | {p['memory_recovery_total']} |", "",
              "这是条目数量审计，不是字节存储、RSS或运行速度对比。轨迹观测与单个记忆代表的字段不同，"
              "不能从条目数或输出数直接推导内存/速度优势。当前冻结CALR代码仍保存支持帧号集合，"
              "本实验不据此宣称支持历史字节存储严格有界。", "",
              "## 6. 场景级配对bootstrap", "",
              f"B={result['protocol']['bootstrap_replicates']}，seed={result['protocol']['bootstrap_seed']}；百分位95%区间，AP点。", "",
              "| 对比 | IoU | 差值 | 95% CI |",
              "|---|---:|---:|---:|"]
    for name, by_threshold in result["bootstrap"].items():
        for t, row in by_threshold.items():
            lines.append(f"| {name} | {t} | {row['point_ap']:+.4f} | [{row['ci95_ap'][0]:+.4f}, {row['ci95_ap'][1]:+.4f}] |")
    lines += ["", "## 7. 独立运行来源", "",
              "三组结果不是同一份预测文件。main_old对应旧主表，canonical对应规范组装/简单记忆表，"
              "prefix_replay使用另一次原生逐帧缓存和冻结辅助缓存。已实测的文件差异见下表。", "",
              "| 运行 | 原生输出 | Full输出 |",
              "|---|---:|---:|"]
    for name, c in result["run_origin"]["counts"].items():
        lines.append(f"| {name} | {c['native']} | {c['full']} |")
    lines += ["", "预测来源目录（具体逐文件SHA256见results.json）：", ""]
    for name, path in result["run_origin"]["roots"].items():
        lines.append(f"- {name}: `{path}`")
    lines += ["", "| 对比 | 原生数量不同的场景 | 原生几何逐行完全相同的场景 | Full完全相同的场景 |",
              "|---|---:|---:|---:|"]
    for name, row in result["run_origin"]["comparisons"].items():
        lines.append(f"| {name} | {row['native_count_different_scenes']} | {row['native_geometry_exact_scenes']} | {row['full_exact_scenes']} |")
    lines += ["", "原生记录内容和输出差异已有文件级证据；仅凭结果文件不能进一步将差异归因于随机采样、CUDA非确定性或异步调度。"
              "本次新增两项实验不重跑前端，所有对比均在同一冻结缓存内配对。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/scannet_recovery_origin_memory_budget_20261029")
    args = parser.parse_args()
    out = args.output
    if (out/"results.json").exists():
        raise RuntimeError("refusing to overwrite completed results")
    protocol_path = out/"protocol.json"
    protocol = json.loads(protocol_path.read_text())
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines()
              if s.strip() and not s.startswith("#")]
    assert len(scenes) == len(set(scenes)) == 100
    config = yaml.safe_load((ROOT/protocol["state_config"]).read_text())["online_candidate_map"]["state"]
    config = {**config, "mvsr_only": False, "selected_calr_v1": True,
              "audit_controls": False, "audit_direct_plr": False,
              "audit_raw_iou_max": False, "audit_mvsr_factors": False,
              "audit_plr_controls": False}
    config["m2"] = {**config["m2"], "strength_variant": "no_visibility", "support_mode": "max"}
    gts, alignments, transform = core.scannet_inputs()
    terminal = {name: {} for name in (
        "base_f000", "full_f000", "base_f015", "full_f015", "base_f030", "full_f030",
        *[f"memory_a{cap}" for cap in protocol["memory_anchor_output_caps"]],
    )}
    terminal_counts = {arm: Counter() for arm in terminal}
    prefixes = {str(f): {arm: {} for arm in ("base", "full", *[
        f"memory_a{cap}" for cap in protocol["memory_anchor_output_caps"]])}
        for f in protocol["prefix_fractions"]}
    prefix_counts = {f: {arm: Counter() for arm in arms} for f, arms in prefixes.items()}
    peaks, per_scene, per_gt, input_hashes = {}, {}, [], {}
    started = time.perf_counter()
    for scene_index, scene in enumerate(scenes, 1):
        npath = ROOT/protocol["native_prefix_root"]/f"{scene}.npz"
        epath = ROOT/protocol["evidence_root"]/f"{scene}.npz"
        native_frames, evidence_frames = core.load_native(npath), core.load_evidence(epath)
        assert [x[0] for x in native_frames] == [x[0] for x in evidence_frames]
        full = core.build_reliability_candidate_map(scene, config)
        proposal = core.ProposalScoreMemory(
            active_capacity=protocol["memory_proposal_active_capacity"],
            archive_capacity=protocol["memory_proposal_archive_capacity"],
            ttl_keyframes=protocol["memory_proposal_ttl_keyframes"],
            output_capacity=protocol["memory_proposal_output_cap"],
        )
        anchor = core.AnchorScoreMemory(capacity=protocol["memory_anchor_state_capacity"],
                                       output_capacity=max(protocol["memory_anchor_output_caps"]))
        checkpoints = {int(np.ceil(len(native_frames)*f))-1: str(f) for f in protocol["prefix_fractions"]}
        trace, scene_peaks = [], {}
        for ordinal, ((frame, native), (_, evidence)) in enumerate(zip(native_frames, evidence_frames)):
            full.update(frame, native, evidence)
            proposal.update(ordinal, frame, evidence.proposal_ids, evidence.proposal_corners, evidence.proposal_scores)
            anchor.update(ordinal, frame, evidence.anchor_ids, evidence.anchor_corners, evidence.anchor_scores)
            valid = np.all(np.ptp(native.corners, axis=1) >= .30, axis=1)
            base = [(0, box.copy(), float(score)) for box, score in zip(native.corners[valid], native.scores[valid])]
            full_snapshot = full.materialize_terminal(frame, native.ids[valid], native.corners[valid], native.scores[valid])
            full_rows = rows_from_snapshot(full_snapshot)
            proposals, anchors = proposal.rows(base), anchor.rows(scene)
            native_hcsr = full_rows[:len(base)]
            outputs = {"base": base, "full": full_rows, **{
                f"memory_a{cap}": native_hcsr+proposals+anchors[:cap]
                for cap in protocol["memory_anchor_output_caps"]}}
            counts = {"base": {"native": len(base), "proposal": 0, "anchor": 0, "total": len(base)},
                      "full": lengths(full_snapshot), **{
                          f"memory_a{cap}": {"native": len(base), "proposal": len(proposals),
                                              "anchor": len(anchors[:cap]), "total": len(outputs[f'memory_a{cap}'])}
                          for cap in protocol["memory_anchor_output_caps"]}}
            state = state_counts(full, proposal, anchor)
            update_peaks(peaks, state)
            update_peaks(scene_peaks, state)
            trace.append({"ordinal": ordinal, "frame_id": frame,
                          "state_counts": state, "output_counts": counts})
            if ordinal in checkpoints:
                f = checkpoints[ordinal]
                tag = f"p{int(float(f)*100)}"
                for arm, rows in outputs.items():
                    prefixes[f][arm][scene] = core.prediction(rows, transform, alignments[scene])
                    prefix_counts[f][arm].update(counts[arm])
                    core.write_prediction(out/"predictions"/f"{arm}_{tag}"/f"{scene}_boxes.pkl", rows)
            if ordinal != len(native_frames)-1:
                continue
            filtered_snapshot = full_snapshot
            for minimum in protocol["native_min_extent_m"]:
                keep = np.all(np.ptp(native.corners, axis=1) >= minimum, axis=1)
                current_base = [(0, box.copy(), float(score)) for box, score in zip(native.corners[keep], native.scores[keep])]
                snapshot = full.materialize_terminal(frame, native.ids[keep], native.corners[keep], native.scores[keep])
                current_full = rows_from_snapshot(snapshot)
                filter_tag = f"f{int(round(minimum*100)):03d}"
                for arm, rows in ((f"base_{filter_tag}", current_base), (f"full_{filter_tag}", current_full)):
                    terminal[arm][scene] = core.prediction(rows, transform, alignments[scene])
                    terminal_counts[arm].update(
                        {"native": len(current_base), "proposal": 0, "anchor": 0, "total": len(current_base)}
                        if arm.startswith("base") else lengths(snapshot))
                    core.write_prediction(out/"predictions"/arm/f"{scene}_boxes.pkl", rows)
            for cap in protocol["memory_anchor_output_caps"]:
                arm = f"memory_a{cap}"
                terminal[arm][scene] = prefixes["1.0"][arm][scene]
                terminal_counts[arm].update(counts[arm])
            n = len(base)
            p = filtered_snapshot.sources.count("m1p")
            source_boxes = {
                "native_internal": native.corners,
                "native_output": native.corners[valid],
                "plr": filtered_snapshot.boxes[n:n+p],
                "calr": filtered_snapshot.boxes[n+p:],
            }
            maximum = {key: maximum_overlap(transform(boxes, alignments[scene]), gts[scene])
                       for key, boxes in source_boxes.items()}
            for gt_index in range(len(gts[scene])):
                per_gt.append({"scene": scene, "gt_index": gt_index, **{
                    f"max_iou_{key}": float(values[gt_index]) for key, values in maximum.items()}})
        per_scene[scene] = {"keyframes": len(native_frames), "peaks": scene_peaks,
                            "keyframe_trace": trace}
        for path in (npath, epath):
            input_hashes[str(path)] = core.sha256(path)
        if scene_index % 10 == 0:
            print(f"replayed {scene_index}/100", flush=True)
    elapsed = time.perf_counter()-started
    assert sum(row["keyframes"] for row in per_scene.values()) == 6817
    prefix_ap = {f: {arm: {str(t): core.class_agnostic_ap(values, gts, t)["ap"]
                          for t in core.THRESHOLDS}
                     for arm, values in arms.items()} for f, arms in prefixes.items()}
    terminal_ap = {arm: {str(t): core.class_agnostic_ap(values, gts, t)["ap"]
                         for t in core.THRESHOLDS} for arm, values in terminal.items()}
    # Reproduce the earlier identical-cache experiment rather than mix runs.
    old = json.loads((ROOT/"reports/scannet_fixed_capacity_prefix_memory_20261014/results.json").read_text())
    for f in prefixes:
        for new_arm, old_arm in (("base", "base"), ("full", "full_recar"), ("memory_a640", "fixed_memory")):
            for t in core.THRESHOLDS:
                assert abs(prefix_ap[f][new_arm][str(t)]-old["exact_ap"][f][old_arm][str(t)]) < 1e-9
    origin = {}
    for t in core.THRESHOLDS:
        categories = {name: Counter() for name in ("internal_native_miss", "removed_by_readout_filter")}
        total_missing = 0
        for row in per_gt:
            if row["max_iou_native_output"] > t:
                continue
            total_missing += 1
            category = "removed_by_readout_filter" if row["max_iou_native_internal"] > t else "internal_native_miss"
            counter = categories[category]
            plr, calr = row["max_iou_plr"] > t, row["max_iou_calr"] > t
            counter.update({"gt": 1, "plr": int(plr), "calr": int(calr), "either": int(plr or calr),
                            "both": int(plr and calr)})
        assert total_missing == sum(c["gt"] for c in categories.values())
        origin[str(t)] = {k: {x: c[x] for x in ("gt", "plr", "calr", "either", "both")}
                         for k, c in categories.items()}
    with (out/"per_gt_recovery_origin.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_gt[0]))
        writer.writeheader(); writer.writerows(per_gt)
    run_origin = compare_runs(scenes, out/"predictions")
    comparisons = {f"full-minus-memory{cap}": ("full_f030", f"memory_a{cap}")
                   for cap in protocol["memory_anchor_output_caps"]}
    comparisons.update({f"full-minus-base-{tag}": (f"full_{tag}", f"base_{tag}")
                        for tag in ("f030", "f015", "f000")})
    result = {"protocol": protocol, "protocol_sha256": core.sha256(protocol_path),
              "scene_count": len(scenes), "gt_count": len(per_gt), "keyframes": 6817,
              "replay_seconds": elapsed, "terminal_ap": terminal_ap,
              "terminal_counts": {a: dict(c) for a, c in terminal_counts.items()},
              "prefix_ap": prefix_ap,
              "prefix_counts": {f: {a: dict(c) for a, c in arms.items()} for f, arms in prefix_counts.items()},
              "recovery_origin": origin, "state_peaks": peaks, "per_scene": per_scene,
              "run_origin": run_origin, "input_sha256": input_hashes,
              "verification": {"matches_20261014_same_cache_AP_to_1e9": True,
                               "new_model_inference": False, "per_scene_terminal_quota_used": False}}
    (out/"point_results.json").write_text(json.dumps(result, indent=2)+"\n")
    print("point estimates saved; starting paired scene bootstrap", flush=True)
    ci = bootstrap(terminal, gts, scenes, comparisons, protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
    for name, (treatment, base) in comparisons.items():
        for t in core.THRESHOLDS:
            ci[name][str(t)]["point_ap"] = terminal_ap[treatment][str(t)]-terminal_ap[base][str(t)]
    result["bootstrap"] = ci
    (out/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"REPORT.md").write_text(make_report(result), encoding="utf-8")
    print(make_report(result), flush=True)


if __name__ == "__main__":
    main()
