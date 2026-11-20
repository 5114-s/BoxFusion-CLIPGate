#!/usr/bin/env python3
"""Add score-only controls, verify file provenance and write paper-ready prose."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_recovery_origin_memory_budget_audit as audit
from tools.bootstrap_final_recar_pairs import scene_pairs, ap_from_pairs


def exact_unordered_comparison(left, right, scenes):
    summary = {"exact_native_geometry_sets": 0,
               "count_mismatch_scenes": [], "matched_pairs": 0,
               "corner_set_error_median_m": None, "corner_set_error_p95_m": None,
               "corner_set_error_max_m": None}
    errors = []
    for scene in scenes:
        a = audit.read_rows(left/f"{scene}_boxes.pkl")
        b = audit.read_rows(right/f"{scene}_boxes.pkl")
        if len(a) != len(b):
            summary["count_mismatch_scenes"].append({"scene": scene, "left": len(a), "right": len(b)})
        if not a or not b:
            summary["exact_native_geometry_sets"] += int(not a and not b)
            continue
        aa = np.stack([x[1] for x in a])
        bb = np.stack([x[1] for x in b])
        distances = np.linalg.norm(aa[:, None, :, None]-bb[None, :, None, :], axis=-1)
        cost = np.maximum(distances.min(3).max(2), distances.min(2).max(2))
        row, col = linear_sum_assignment(cost)
        current = cost[row, col]
        errors.extend(current.tolist())
        summary["matched_pairs"] += len(current)
        summary["exact_native_geometry_sets"] += int(len(a) == len(b) and np.all(current == 0))
    summary.update({f"corner_set_error_{name}_m": float(fn(errors)) for name, fn in (
        ("median", np.median), ("p95", lambda x: np.percentile(x, 95)), ("max", np.max))})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT/"reports/scannet_recovery_origin_memory_budget_20261029")
    parser.add_argument("--render-only", action="store_true",
                        help="Regenerate prose from saved statistics without replay or bootstrap")
    args = parser.parse_args()
    out = args.output
    result = json.loads((out/"results.json").read_text())
    if args.render_only:
        render_report(out, result, json.loads((out/"secondary_controls_and_validation.json").read_text()))
        return
    protocol = result["protocol"]
    scenes = [s.strip() for s in (ROOT/protocol["scene_list"]).read_text().splitlines()
              if s.strip() and not s.startswith("#")]
    gts, alignment, transform = audit.core.scannet_inputs()
    predictions = {}
    # Secondary decomposition: no new data-dependent budget or parameter.
    # Take the fixed native prefix of each already evaluated Full output.
    for tag in ("f030", "f015", "f000"):
        for arm in (f"full_{tag}", f"hcsr_{tag}"):
            predictions[arm] = {}
        for scene in scenes:
            filename = f"{scene}_boxes.pkl"
            base = audit.read_rows(out/"predictions"/f"base_{tag}"/filename)
            full = audit.read_rows(out/"predictions"/f"full_{tag}"/filename)
            native_hcsr = full[:len(base)]
            assert len(native_hcsr) == len(base)
            assert all(np.array_equal(x[1], y[1]) for x, y in zip(native_hcsr, base))
            audit.core.write_prediction(out/"predictions"/f"hcsr_{tag}"/filename, native_hcsr)
            for arm, rows in ((f"hcsr_{tag}", native_hcsr), (f"full_{tag}", full)):
                predictions[arm][scene] = audit.core.prediction(rows, transform, alignment[scene])
    exact = {arm: {str(t): audit.core.class_agnostic_ap(values, gts, t)["ap"]
                   for t in audit.core.THRESHOLDS} for arm, values in predictions.items()}
    comparisons = {f"recovery-increment-{tag}": (f"full_{tag}", f"hcsr_{tag}")
                   for tag in ("f030", "f015", "f000")}
    ci = audit.bootstrap(predictions, gts, scenes, comparisons,
                         protocol["bootstrap_replicates"], protocol["bootstrap_seed"])
    for name, (treatment, base) in comparisons.items():
        for t in audit.core.THRESHOLDS:
            ci[name][str(t)]["point_ap"] = exact[treatment][str(t)]-exact[base][str(t)]

    # Verify deterministic bootstrap tie convention against the point estimator.
    tie_audit = {}
    for arm in result["terminal_ap"]:
        directory = out/"predictions"/(f"{arm}_p100" if arm.startswith("memory") else arm)
        values = {scene: audit.core.prediction(audit.read_rows(directory/f"{scene}_boxes.pkl"),
                                              transform, alignment[scene]) for scene in scenes}
        tie_audit[arm] = {}
        for t in audit.core.THRESHOLDS:
            pairs = np.concatenate([scene_pairs(*values[s], gts[s], t) for s in scenes])
            deterministic = ap_from_pairs(pairs, sum(len(gts[s]) for s in scenes))
            point = result["terminal_ap"][arm][str(t)]
            tie_audit[arm][str(t)] = {"exact_ap": point, "deterministic_ap": deterministic,
                                     "absolute_delta_ap": abs(point-deterministic)}
    maximum_tie_delta = max(row["absolute_delta_ap"] for arms in tie_audit.values() for row in arms.values())
    assert maximum_tie_delta < .05, maximum_tie_delta

    origin_roots = result["run_origin"]["roots"]
    geometry_audit = {
        "main_old_vs_canonical": exact_unordered_comparison(
            Path(origin_roots["main_old"])/"base", Path(origin_roots["canonical"])/"base", scenes),
        "canonical_vs_prefix_replay": exact_unordered_comparison(
            Path(origin_roots["canonical"])/"base", out/"predictions/base_p100", scenes),
    }
    summary = {"posthoc_score_only_decomposition": True,
               "no_new_parameter_or_budget_selection": True,
               "hcsr_only_exact_ap": exact, "recovery_increment_bootstrap": ci,
               "bootstrap_tie_validation": tie_audit,
               "maximum_point_vs_deterministic_tie_delta_ap": maximum_tie_delta,
               "unordered_native_geometry_audit": geometry_audit}
    (out/"secondary_controls_and_validation.json").write_text(json.dumps(summary, indent=2)+"\n")
    render_report(out, result, summary)
    print(json.dumps({"hcsr_only": exact, "recovery_increments": ci,
                      "maximum_tie_delta": maximum_tie_delta,
                      "geometry_matching": geometry_audit}, indent=2), flush=True)


def render_report(out, result, summary):
    exact = summary["hcsr_only_exact_ap"]
    ci = summary["recovery_increment_bootstrap"]
    maximum_tie_delta = summary["maximum_point_vs_deterministic_tie_delta_ap"]
    geometry_audit = summary["unordered_native_geometry_audit"]

    sections = ["", "## 8. 恢复增益与HCSR增益分离（追加诊断）", "",
                "从已评估Full输出中直接读取原生前缀作为HCSR-only；几何、分数和HCSR历史不变。"
                "本项为主审计完成后的追加诊断，未新增或修改预算和参数。Full−HCSR-only隔离恢复记录的增量。", "",
                "| 原生最小尺寸 | HCSR-only AP15/AP25/AP50 | Full−HCSR-only | 95%区间（依次三阈值） |",
                "|---:|---|---|---|"]
    for tag, size in (("f030", .3), ("f015", .15), ("f000", .0)):
        h = exact[f"hcsr_{tag}"]
        increment = ci[f"recovery-increment-{tag}"]
        ap_text = "/".join(f"{h[str(t)]:.4f}" for t in audit.core.THRESHOLDS)
        delta = "/".join(f"{increment[str(t)]['point_ap']:+.4f}" for t in audit.core.THRESHOLDS)
        interval = "; ".join(f"[{increment[str(t)]['ci95_ap'][0]:+.4f},{increment[str(t)]['ci95_ap'][1]:+.4f}]"
                             for t in audit.core.THRESHOLDS)
        sections.append(f"| {size:.2f} | {ap_text} | {delta} | {interval} |")
    sections += ["", "## 9. bootstrap与运行文件核验", "",
                 f"点估计采用现有AP评测器；bootstrap采用确定性的场景内和汇总稳定排序破同。"
                 f"全部新增工作点中，两种破同的最大AP差为{maximum_tie_delta:.6f}点（小于0.05点）。"
                 "置信区间为探索性逐项区间，未进行多重比较校正；不据此宣称所有预算上的普遍优势。", "",
                 "运行来源还执行了无序框集合匹配，排除了仅由输出行顺序不同造成的表观差异：", "",
                 "| 对比 | 无序几何集合完全相同的场景 | 匹配框角点集合误差中位数(m) | P95(m) |",
                 "|---|---:|---:|---:|"]
    for name, row in geometry_audit.items():
        sections.append(f"| {name} | {row['exact_native_geometry_sets']} | {row['corner_set_error_median_m']:.8f} | {row['corner_set_error_p95_m']:.8f} |")
    sections += ["", "## 10. 可直接用于正文的文字", "",
                 "**恢复来源（第4.5节）。** 为区分内部原生地图遗漏与读出过滤造成的输出遗漏，我们在同一冻结缓存重放中，"
                 "按每个GT的最大3D AABB IoU审计过滤前后的原生集合及实际PLR/CALR输出。"
                 "IoU为0.25时，608个原生输出遗漏目标中，486个在过滤前也未被覆盖，122个在过滤前已被覆盖但被尺寸读出规则排除。"
                 "两类目标分别有277和109个被至少一个恢复分支覆盖。"
                 "因此，恢复既补充内部地图未覆盖的目标，也补充原生读出限制下的输出遗漏；本文统一将总体对象称为“原生输出遗漏”。"
                 "该统计不考虑分数与一对一匹配，不等同于标准检测召回率。", "",
                 "**尺寸过滤工作点（第4.5节或补充材料）。** 将原生逐轴最小尺寸从0.30 m降至0.15 m或关闭，"
                 "并重新针对对应原生输出执行PLR读出后，完整方法相对Ctrl仍分别提高10.66/10.08/5.97和11.07/10.39/6.08个AP点。"
                 "宽松工作点使两种系统的绝对AP均下降，但完整方法的增益仍存在，说明其总体收益并非仅由0.30 m读出限制产生。"]
    inc = ci["recovery-increment-f000"]
    sections += ["此外，在关闭尺寸过滤的工作点，Full相对HCSR-only仍提高"
                 + "/".join(f"{inc[str(t)]['point_ap']:.2f}" for t in audit.core.THRESHOLDS)
                 + "个AP点，直接说明恢复记录在该工作点仍有额外贡献。", "",
                 "**简单因果记忆预算（第4.4节）。** 在新增评估前固定每场景proposal读出上限12及anchor上限128、256、512，"
                 "并以已有640上限复核，所有路线共享冻结候选、原生记录、输出评分和HCSR。"
                 "256预算点输出28,364条，完整方法输出26,060条；后者AP15/AP25/AP50分别高0.39/0.47/0.22个点。"
                 "其2,000次场景配对bootstrap 95%区间依次为[0.02,0.78]、[0.14,0.84]和[0.01,0.43]。"
                 "但128预算的简单记忆只输出15,787条，比完整方法少39.42%，AP仅低0.57/0.64/0.30个点，"
                 "说明调小简单记忆预算确实可以得到更少输出和接近的AP，不能把完整系统相对Base的大增益全部归因于层级状态。"
                 "在512/640更大预算下，AP15和AP50区间跨零，因此额外收益有限且依赖比较工作点，不能声称层级确认在所有预算下显著更优。", "",
                 "**前缀与状态边界（补充材料）。** 简单记忆在25%前缀仍具有更高AP，完整方法从75%前缀起在所测预算下三项AP均更高。"
                 "完整恢复状态与简单记忆的同时占用条目峰值为1,747和2,171，但字段和历史长度不同，该数值只描述实际条目规模，"
                 "不能直接作为字节内存或运行效率优势的证据。", "",
                 "**独立运行来源（第4.1节）。** 主表47.09/42.61/22.04来自mvsr_design_20261005的原生运行及无visibility分数读出，"
                 "原生输出为1,785条；简单记忆表47.03/42.61/22.14采用另一原生运行的1,787条记录与规范组件组装。"
                 "前缀表47.21/42.72/22.15及本节新增对照使用mvsr_multiview_audit_20261012逐关键帧原生缓存和冻结辅助证据。"
                 "不同组的原生记录数量分布及几何内容存在可核验差异，即使总框数相同也不是同一集合。"
                 "本文只在共享相同预测来源的面板内进行配对比较，未将独立运行差异直接归因于随机性或异步调度。", "",
                 "**旧主表末位核验。** 上段47.09/42.61/22.04为稿件现有标注；本次定位到的旧主表源预测实测为"
                 "47.086718/42.605135/22.034981，按两位小数应为47.09/42.61/22.03。"
                 "AP50末位与稿件相差0.01，应核对最终预测来源，而不是把新旧不同运行强行统一。", "",
                 "**输出集合定义（第3.1和3.5节）。** 建议分别记内部原生地图为M_t、尺寸过滤后的原生输出为N_t=Filter(M_t)，"
                 "HCSR在N_t上保持身份和几何，仅替换分数。PLR在状态更新阶段参考当前内部地图维护覆盖状态，"
                 "实际读出时针对N_t重新检查覆盖并执行分支内NMS和输出上限选择；CALR独立读出，最后直接拼接记录。", "",
                 "**图表与摘要口径。** 主文统一写“原生输出遗漏”；只有本审计第一类可直接称为“内部地图未覆盖”。"
                 "图2确认标注统一为3-keyframe confirmation。效率统一表述为25.27原始输入帧等效FPS、关键帧间隔25，"
                 "避免表述为25.27 Hz对象地图更新。章节交叉引用应按最终文档编号更新。", ""]
    base_report = audit.make_report(result)
    (out/"REPORT.md").write_text(base_report, encoding="utf-8")
    report = base_report+"\n".join(sections)
    (out/"论文可用实验结果.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
