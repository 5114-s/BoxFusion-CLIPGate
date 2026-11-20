#!/usr/bin/env python3
"""Generate paper-ready DOCX and text from the completed duplicate audit."""
from collections import Counter
import csv
import json
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/scannet_cross_branch_duplicates_20261115"


def read_csv(name):
    return list(csv.DictReader((OUT / name).open(encoding="utf-8-sig")))


def main():
    result = json.loads((OUT / "results.json").read_text())
    assert result["validation"]["status"] == "PASS"
    metrics = result["metrics"]
    ts = ("0.15", "0.25", "0.5")
    policies = result["protocol"]["policies"]
    names = {"concat": "直接拼接（当前）", "cross_only_0.25": "仅跨分支 NMS 0.25",
        "cross_only_0.5": "仅跨分支 NMS 0.50", "global_nms_0.25": "统一 NMS 0.25",
        "global_nms_0.5": "统一 NMS 0.50", "native_priority_0.25": "保留全部原生框 NMS 0.25",
        "native_priority_0.5": "保留全部原生框 NMS 0.50"}
    base = metrics["full/concat"]
    d25, d50 = base["0.25"], base["0.5"]
    cross = metrics["full/cross_only_0.25"]
    ci = {(r["policy"], str(r["iou"])): r for r in result["paired_bootstrap"]}
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(1.6)
    section.top_margin = section.bottom_margin = Cm(1.6)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(9.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    text = []

    def heading(value, level=1):
        doc.add_heading(value, level=level)
        text.extend(["", value, ""])

    def para(value):
        doc.add_paragraph(value)
        text.extend([value, ""])

    def table(headers, rows):
        grid = doc.add_table(rows=1, cols=len(headers))
        grid.style = "Table Grid"
        for cell, value in zip(grid.rows[0].cells, headers):
            cell.text = str(value)
        grid.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for row in rows:
            for cell, value in zip(grid.add_row().cells, row):
                cell.text = str(value)
        for row in grid.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.size = Pt(8)
        text.append("\t".join(headers))
        text.extend("\t".join(map(str, row)) for row in rows)
        text.append("")

    f = lambda value: f"{value:.2f}"
    heading("CaRR-3D：跨分支重复检测与 AP 影响实证分析", 0)
    para("实验日期：2026-11-15。ScanNet-100 终态冻结预测，100 场景、1,433 个 GT；"
         "1,787 个原生框 + 1,101 个 PTR 框 + 23,172 个 VCR 框 = 26,060 条预测。"
         "不重新运行模型，不使用 GPU，不修改现有算法、评分或论文。")
    heading("1. 结论与正文最小补充")
    para("跨分支重复确实存在，主要涉及 VCR；评分映射把记录放入低分区间，并不会删除重复。"
         "不使用 GT 的几何 NMS 显著减少重复，但同时删除部分可正确定位的记录：AP15/AP25 提高，AP50 和完整输出召回下降。"
         "因此，应报告覆盖与重复的取舍，不能写成‘重复无影响’或‘统一 NMS 必然更好’。")
    paragraph = (
        "跨分支重复分析。我们固定原生及恢复预测的几何和评分，仅改变最终读出阶段的去重规则。"
        f"在 ScanNet-100 上，IoU 0.25 下共有 {d25['cross_duplicate_fp']:,} 条跨分支重复 FP，"
        f"占全部预测的 {d25['cross_duplicate_rate_percent']:.2f}%。仅跨分支 NMS（阈值 0.25）"
        f"将该数量降至 {cross['0.25']['cross_duplicate_fp']:,}，并使 AP25 从 {d25['AP']:.2f} 提高至 "
        f"{cross['0.25']['AP']:.2f}，但 AP50 从 {d50['AP']:.2f} 降至 {cross['0.5']['AP']:.2f}。"
        "这说明当前拼接包含重复预测，而简单几何抑制会同时损失部分有效检测；直接拼接体现覆盖与去重之间的取舍，"
        "并非对所有评测阈值均最优。完整分支对统计、去重对照与场景级配对区间见补充材料。")
    para("可直接用于正文（不新增独立章节）：")
    para(paragraph)
    main_policies = ("concat", "cross_only_0.25", "cross_only_0.5", "global_nms_0.25", "global_nms_0.5")
    table(["读出策略", "输出数", "跨分支重复FP@.25", "AP15", "AP25", "AP50"], [
        [names[p], metrics['full/'+p]['0.25']['predictions'], metrics['full/'+p]['0.25']['cross_duplicate_fp'],
         *[f(metrics['full/'+p][t]['AP']) for t in ts]] for p in main_policies])
    para("表注：NMS 阈值是预测框之间的三维 AABB IoU 阈值，列中的 @.25 是检测评测阈值；二者不是同一概念。"
         "所有策略共享同一预测缓存；保留框的几何、分数与相对装配顺序不变。AP 为百分数，差值以 AP 点计。"
         "主文可仅留前三行；完整统一 NMS 与原生优先结果放补充材料。")

    heading("2. 指标定义与公平控制")
    para("最终读出：HSR 更新原生记录分数后，与 PTR、VCR 记录直接拼接。PTR 的原生覆盖排除和分支内部 NMS "
         "是既有分支规则，不等于最终输出执行统一跨分支去重。这里的消融额外作用于冻结的终态拼接结果，"
         "不反馈建图或修改任何分支状态。旧缓存中的 proposal/anchor 标签在本报告分别对应 PTR/VCR，"
         "anchor 是历史代码字段，不改变当前 pre-NMS candidates 术语。")
    para("检测匹配沿用当前评测器：稳定降序排序，预测取最大重叠 GT，IoU 严格大于阈值且该 GT 未匹配时为 TP；"
         "同一 GT 已被更高排名预测占用的后续记录计为重复 FP。若其来源与第一条 TP 相同，归为分支内重复；"
         "否则归为跨分支重复。每条重复 FP 只计一次，按第一条 TP 来源归因，因此分支对统计依赖评分顺序。"
         "重复是 FP 的子集，TP、FP、重复不能作为三类互斥总数相加。")
    para("几何重叠诊断与检测重复分开：前者计算某分支有多少框与另一分支至少一个框相交超过阈值，"
         "并不等于重复目标数。prediction multiplicity 统计每个 GT 被多少预测覆盖，其均值以全部 1,433 个 GT 为分母；"
         "新增唯一 GT 覆盖也是最大 IoU 诊断，不称为标准检测召回率。标准 Recall 是一对一 TP/GT。")
    para("所有 NMS 使用原世界坐标的 AABB，与既有实现一致；GT 匹配则采用统一 ScanNet 对齐和轴变换。"
         "仅跨分支 NMS 按分数降序，仅对已保留的不同来源记录进行抑制；统一 NMS 同时处理分支内与跨分支。"
         "原生优先 NMS 保留全部原生记录，再对辅助记录执行抑制。三种策略均不读取 GT。"
         "阈值 0.25、0.50 沿用既有诊断，不根据本次结果再调参；该分析为已研究测试集上的探索性诊断。")

    heading("3. 哪些分支产生重复")
    duplicates = read_csv("full_duplicate_records.csv")
    order = {"Native": 0, "PTR": 1, "VCR": 2}
    pair_counts = Counter((r['evaluation_iou'], tuple(sorted((r['first_tp_branch'], r['branch']), key=order.get))) for r in duplicates)
    pairs = [("Native", "PTR"), ("Native", "VCR"), ("PTR", "VCR"),
             ("Native", "Native"), ("PTR", "PTR"), ("VCR", "VCR")]
    table(["重复 FP 来源对", "IoU 0.15", "IoU 0.25", "IoU 0.50"], [
        ["–".join(p), *[pair_counts[t, p] for t in ts]] for p in pairs] + [
        ["跨分支合计", *[base[t]['cross_duplicate_fp'] for t in ts]],
        ["分支内合计", *[base[t]['within_duplicate_fp'] for t in ts]],
        ["全部 FP", *[base[t]['fp'] for t in ts]]])
    table(["评测IoU", "TP", "FP", "跨分支重复占预测%", "标准Recall%", "平均multiplicity", "新增唯一GT覆盖"], [
        [t, base[t]['tp'], base[t]['fp'], f(base[t]['cross_duplicate_rate_percent']),
         f(base[t]['recall_percent']), f(base[t]['mean_multiplicity_all_gt']), base[t]['additional_unique_gt_coverage']] for t in ts])
    para("Native–PTR 的检测重复数量不必为零：‘两个预测分别与同一 GT 的 IoU 达标’不意味着它们彼此的 IoU "
         "也超过 PTR 的覆盖排除阈值。不能仅凭此表判定 PTR 原生覆盖检查没有执行。")
    geometry = Counter()
    denominators = Counter()
    for r in read_csv("geometric_overlap_per_scene.csv"):
        key = (r['nms_iou'], r['reference'], r['tested'])
        geometry[key] += int(r['overlapping_records'])
        denominators[key] += int(r['tested_count'])
    table(["参考分支→被检查分支", "IoU≥.25 框数/总数", "比例%", "IoU≥.50 框数/总数", "比例%"], [
        [a+'→'+b, f"{geometry['0.25', a, b]}/{denominators['0.25', a, b]}",
         f(100*geometry['0.25',a,b]/max(1,denominators['0.25',a,b])),
         f"{geometry['0.5', a, b]}/{denominators['0.5', a, b]}",
         f(100*geometry['0.5',a,b]/max(1,denominators['0.5',a,b]))]
        for a in order for b in order if a != b])
    para("上表分母为被检查分支的全部输出数，同一预测可同时重叠多个分支，不可跨行求和当作总重复数。")

    heading("4. 去重的收益、损失与配对不确定性")
    table(["策略", "保留框", "删除Native", "重复FP@.25 跨/内", "Recall25%", "Recall50%"], [
        [names[p], metrics['full/'+p]['0.25']['predictions'], metrics['full/'+p]['0.25']['removed_native'],
         f"{metrics['full/'+p]['0.25']['cross_duplicate_fp']}/{metrics['full/'+p]['0.25']['within_duplicate_fp']}",
         f(metrics['full/'+p]['0.25']['recall_percent']), f(metrics['full/'+p]['0.5']['recall_percent'])] for p in policies])
    table(["策略−当前拼接", "ΔAP15 [95% CI]", "ΔAP25 [95% CI]", "ΔAP50 [95% CI]"], [
        [names[p], *[f"{ci[p,t]['delta_AP_vs_concat']:+.2f} [{ci[p,t]['ci95_low']:+.2f}, {ci[p,t]['ci95_high']:+.2f}]" for t in ts]]
        for p in policies if p != 'concat'])
    para("区间：100 场景成对 bootstrap，2,000 次、种子 20261115，每次重采样后重新池化计算 AP；"
         "报告未作多重比较校正的边际百分位区间，不作为独立验证集的确认性显著结论。")
    removal = Counter()
    for r in read_csv("removed_prediction_accounting.csv"):
        for key in ('removed_old_tp', 'removed_old_within_dup', 'removed_old_cross_dup', 'removed_old_other_fp', 'new_tp_reassigned_after_removal'):
            removal[r['policy'], r['iou'], key] += int(r[key])
    table(["策略（IoU .50评测）", "删除原TP", "删除原跨分支重复", "剩余重复重新成为TP", "净TP损失"], [
        [names[p], removal[p,'0.5','removed_old_tp'], removal[p,'0.5','removed_old_cross_dup'],
         removal[p,'0.5','new_tp_reassigned_after_removal'], d50['tp'] - metrics['full/'+p]['0.5']['tp']]
        for p in main_policies if p != 'concat'])
    para("删除框的身份按原始拼接匹配标记；删除原 TP 后，剩余重复记录可能重新成为 TP，所以删除 TP 数不等于净 TP 损失。"
         "即便去重后未删除原生框，也可能删除几何更准确的辅助框，不能把原生保留等同于准确率不受影响。")

    heading("5. 同规则对照与有限输出")
    method_names = {"base": "同缓存 Ctrl", "memory256": "同缓存简单记忆12/256", "full": "CaRR-3D"}
    table(["方法", "读出规则", "输出数", "AP15", "AP25", "AP50"], [
        [method_names[m], names[p], metrics[m+'/'+p]['0.25']['predictions'], *[f(metrics[m+'/'+p][t]['AP']) for t in ts]]
        for m in method_names for p in ('concat', 'cross_only_0.25', 'global_nms_0.25')])
    table(["策略", "R@50 IoU.25%", "R@50 IoU.50%", "R@100 IoU.25%", "R@100 IoU.50%"], [
        [names[p], *[f(metrics['full/'+p][t][k]) for k,t in
                    (('R50_percent','0.25'), ('R50_percent','0.5'), ('R100_percent','0.25'), ('R100_percent','0.5'))]] for p in main_policies])
    para("同缓存 Ctrl 是受控原生路径，不是原论文 BoxFusion 的外部方法行。简单记忆使用相同 HSR 与冻结候选缓存，"
         "用于核查不能只给 CaRR-3D 添加有利后处理。完整三方法×七策略及逐场景结果见 CSV。R@K 为各场景独立 Top-K 后，汇总 TP/全部 GT。")

    heading("6. 重复 FP 本身造成多少 AP 损失（仅 GT-oracle 诊断）")
    oracle = read_csv('oracle_duplicate_only_AP.csv')
    table(["评测IoU", "当前 AP", "仅删跨分支重复 AP", "增加AP点", "仅删全部重复 AP"], [
        [t, f(base[t]['AP']),
         *[f(float(next(r for r in oracle if r['iou']==t and r['arm']=='oracle_remove_cross_duplicate_fp')[k])) for k in ('AP','delta_AP')],
         f(float(next(r for r in oracle if r['iou']==t and r['arm']=='oracle_remove_all_duplicate_fp')['AP']))] for t in ts])
    para("此诊断使用测试 GT 识别并仅删除重复 FP，所有原 TP、分数和排序均保留，所以 Recall 不变。"
         "它用于量化现有重复 FP 的 AP 代价，不是可以部署的算法，也不是公平方法比较。不同评测 IoU 对应不同删除集合，"
         "不能将其三项 AP 当作同一组预测的改进结果，更不能据此声称已有去重算法达到该值。")

    heading("7. 论文安排与可追溯性")
    para("建议正文：在输出装配处保留‘直接拼接，不执行额外统一 NMS’的一句说明；实验部分加入第1节简短段落和三行表。"
         "补充材料：第3–6节完整统计、阈值敏感性与配对区间。无需新增数据集或重跑检测器。"
         "这些结果支持‘已量化重复与覆盖的取舍’，不支持‘跨分支重复已被消除’、‘评分映射解决重复’或‘不做 NMS 全面更优’。")
    para("当前表基线 AP15/AP25/AP50 = 47.2222425543 / 42.7416249707 / 22.1578968805，"
         "与 scannet_always_aux_factorial_20261102 的 Full 行一致；不要混入旧主运行 47.09 或旧非稳定同分排序 47.21 的数字。"
         "原始预测、场景列表和评测代码均记录 SHA256，已核验原文件没有变化。全部保留记录的几何、评分和装配顺序保持不变。")
    para("复现入口：tools/analyze_cross_branch_duplicates_20261115.py（boxfusion-online 环境，CUDA_VISIBLE_DEVICES=''）；"
         "tools/diagnose_duplicate_only_ap_20261115.py 为独立 oracle 诊断；本报告由 tools/report_cross_branch_duplicates_20261115.py 生成。"
         "结果目录含 protocol.json、input_sha256.json、results.json、metrics.csv、paired_bootstrap.csv、"
         "full_duplicate_records.csv、per_scene_metrics.csv、per_branch_metrics.csv、removed_prediction_accounting.csv、"
         "geometric_overlap_per_scene.csv 和 oracle_duplicate_only_AP.csv。")
    doc.save(OUT / '跨分支重复检测_实证分析与论文补充.docx')
    (OUT / '实验结果与正文建议.txt').write_text('\n'.join(text), encoding='utf-8')
    print(OUT / '跨分支重复检测_实证分析与论文补充.docx')


if __name__ == '__main__':
    main()
