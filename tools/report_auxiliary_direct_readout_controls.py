#!/usr/bin/env python3
"""Write a concise Chinese DOCX/plain-text report from completed JSON results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "reports/scannet_auxiliary_direct_readout_20261102"
LABELS = {
    "base": "Ctrl：原生分数，无恢复",
    "hcsr": "HCSR：无恢复",
    "plr": "HCSR＋PLR",
    "calr": "HCSR＋CALR",
    "full": "Full：HCSR＋PLR＋CALR",
    "memory256": "HCSR＋持久简单记忆12/256（既有）",
    "current_p_all": "HCSR＋当前proposal全部直接输出",
    "current_p12": "HCSR＋当前proposal Top-12",
    "current_a_all": "HCSR＋当前anchor全部直接输出",
    "current_both_all": "HCSR＋当前proposal/anchor全部",
    "current_p12_a_all": "HCSR＋当前proposal Top-12/全部anchor",
    "replace_plr_current12": "仅将PLR替换为当前proposal Top-12",
    "replace_calr_current_all": "仅将CALR替换为当前全部anchor",
    "current_p_all_raw": "HCSR＋当前proposal全部（原始评分）",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    out = args.output
    result = json.loads((out/"results.json").read_text())
    assert result["validation"]["status"] == "PASS"
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin = section.right_margin = Cm(1.5)
    section.top_margin = section.bottom_margin = Cm(1.5)
    normal = doc.styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(9)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    text = []

    def paragraph(value):
        doc.add_paragraph(value)
        text.append(value+"\n")

    def heading(value, level=1):
        doc.add_heading(value, level=level)
        text.append("\n"+value+"\n")

    def table(headers, rows):
        grid = doc.add_table(rows=1, cols=len(headers))
        grid.style = "Table Grid"
        for cell, value in zip(grid.rows[0].cells, headers):
            cell.text = value
        repeat = OxmlElement("w:tblHeader")
        grid.rows[0]._tr.get_or_add_trPr().append(repeat)
        for row in rows:
            for cell, value in zip(grid.add_row().cells, row):
                cell.text = str(value)
        for row in grid.rows:
            for cell in row.cells:
                for para in cell.paragraphs:
                    for run in para.runs:
                        run.font.size = Pt(8)
        text.append("\t".join(headers))
        text.extend("\t".join(map(str, row)) for row in rows)
        text.append("")

    heading("ScanNet-100：辅助候选直接读出消融", 0)
    paragraph("实验日期：2026-11-02。100场景，1,433个GT，6,817个关键帧，终态共享1,787个原生框。"
              "使用既有冻结候选和原生状态缓存，不重新运行模型、不使用GPU，也不修改生产推理代码。")
    heading("1. 比较内容与边界")
    paragraph("本实验回答：保留相同WeDetect-Uni/Boxer辅助前端及HCSR时，跳过PLR/CALR状态维护，"
              "直接输出当前关键帧提升后三维候选，是否能够替代历史恢复。除Ctrl行外，所有路线共享完全相同的原生几何、"
              "记录顺序和三因素无visibility历史最大HCSR分数。PLR、CALR及完整参考输出来自同一缓存的因果前缀导出，"
              "并逐前缀核对既有单因素实验和AP。")
    paragraph("当前直接读出没有跨关键帧候选状态：新关键帧到达后，上一帧直接输出的辅助框被替换，空候选帧会清空该分支。"
              "不做轨迹关联、体素聚合、三帧确认、medoid、原生覆盖排除或辅助三维NMS。proposal读取全部或预设Top-12；"
              "anchor读取缓存中的全部300个。post-NMS proposal是检测器通常的输出，pre-NMS anchor是另行保留的中间候选。")
    paragraph("主比较使用原系统评分：proposal按相同尺寸分桶赋分，anchor使用相同确定性尾部分数映射；"
              "另设一条proposal原始分数诊断。所有预算均为预设上限，实际输出数没有强制相等。"
              "CALR可保存640个历史确认体素，当前anchor最多300个，因此不能将此实验解释为同数量、同状态或同内存对照。")
    paragraph("该数据集此前已经用于实验与分析。本次属于事后探索性验证，不是独立盲测。"
              "最终关键帧只对应局部视角；本实验只能检验当前候选能否替代历史恢复，不能单独证明轨迹/体素组织优于持久简单记忆。")

    heading("2. 终态AP与实际辅助输出数量")
    paragraph("下表主报确定性稳定排序AP。预测按原生、proposal、anchor拼接；严格IoU>阈值，一对一最高IoU匹配。"
              "完整方法与旧主表47.09/42.61/22.04属于不同运行来源，不可交叉计算增益。")
    terminal = result["prefix_results"]["1.0"]
    rows = []
    for arm, label in LABELS.items():
        value = terminal[arm]
        counts, ap = value["counts"], value["stable_ap"]
        rows.append([label, counts.get("proposal", 0), counts.get("anchor", 0),
                     *[f"{ap[str(t)]:.4f}" for t in (.15, .25, .5)]])
    table(["路线", "proposal数", "anchor数", "AP15", "AP25", "AP50"], rows)
    paragraph("所有路线另外包含1,787个原生框。memory256为既有持久简单记忆缓存的同协议复核，"
              "不是新运行、不是当前帧无状态路线。完整旧评测器AP与稳定排序AP同时保存在prefix_ap.csv，"
              "避免把同分破同差异误认为方法变更。")

    heading("3. 场景级配对bootstrap")
    paragraph("固定种子20261102，2,000次场景级配对重采样。以下为AP点差及边际95%百分位区间，"
              "属于探索性比较，未作多重比较校正。点差与区间均使用稳定排序AP。")
    ci_rows = []
    for name, by_threshold in result["paired_bootstrap"].items():
        for t in (.15, .25, .5):
            value = by_threshold[str(t)]
            low, high = value["ci95_ap"]
            ci_rows.append([name, f"{t:.2f}", f"{value['point_ap']:+.4f}",
                            f"[{low:+.4f}, {high:+.4f}]"])
    table(["比较", "IoU", "AP差", "95%区间"], ci_rows)

    heading("4. 序列前缀结果")
    paragraph("25%/50%/75%/100%均在对应最后一个已提交关键帧读取输出，使用相同0.30 m原生过滤与输出装配。"
              "100%就是本次终态结果，不另做历史回放或末端额外累计。各前缀以完整场景GT评测，不是仅可见目标AP。")
    rows = []
    for fraction, arms in result["prefix_results"].items():
        for arm in ("hcsr", "current_both_all", "memory256", "full"):
            value = arms[arm]
            rows.append([f"{100*float(fraction):.0f}%", LABELS[arm], value["total"],
                *[f"{value['stable_ap'][str(t)]:.4f}" for t in (.15, .25, .5)]])
    table(["前缀", "路线", "总框数", "AP15", "AP25", "AP50"], rows)

    heading("5. 可以写入论文的结论")
    delta = result["paired_bootstrap"]["Full-minus-current-PA"]
    pieces = [f"{delta[str(t)]['point_ap']:+.2f}" for t in (.15, .25, .5)]
    intervals_positive = all(delta[str(t)]["ci95_ap"][0] > 0 for t in (.15, .25, .5))
    paragraph("在固定辅助候选来源、原生框与历史支持重评分的条件下，完整恢复相对仅输出当前关键帧全部proposal/anchor，"
              "AP15/AP25/AP50分别变化"+"/".join(pieces)+"点。"+
              ("三项场景配对95%区间均高于零。" if intervals_positive else "具体区间见上表，不作统一显著优势断言。")+
              "该对照支持历史恢复优于此处定义的当前帧直接读出，但不把收益进一步归因于某一个确认、关联或代表框规则。")
    paragraph("持久简单记忆仍是必要对照：它同样保存过去视角中的候选，与当前帧直接输出不同。"
              "不能用本实验取代已有强简单记忆比较，也不能据此改写成PLR/CALR各自稳定优于强简单记忆。"
              "当前帧路线不执行原生覆盖排除与三维NMS，完整路线保留原有规则；因此这些结果属于恢复模块整体替代对照，"
              "不是仅改变三帧确认这一单因素的消融。CPU读出耗时不是端到端FPS，本实验没有新增效率结论。")
    heading("6. 完整结果与复现")
    paragraph("results.json：全部前缀、工作点、区间和协议；prefix_ap.csv：14路线×4前缀，含旧/稳定AP；"
              "terminal_output_counts.csv：逐场景输出数；paired_bootstrap.csv：配对区间；finite_workpoints.csv："
              "Top-20/50/100/300与每场1/5/10 FP的诊断；tp_fp_duplicates.csv：TP、FP与分支内/跨分支重复；"
              "pr_*.npz：完整PR曲线；current_keyframe_readout_trace.csv：6,817关键帧当前读出记录；"
              "materialization_manifest.json：输入SHA256及共享组件核对；predictions：全部逐前缀预测。"
              "重复预测属于FP子集，不作为第三类与TP/FP相加。固定FP结果使用测试GT选择整分数阈值，仅供oracle诊断。")
    paragraph("复现脚本：tools/run_auxiliary_direct_readout_controls.py。设置CUDA_VISIBLE_DEVICES为空、"
              "OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1、MKL_NUM_THREADS=1后，使用boxfusion-online环境运行。"
              "脚本拒绝覆盖已完成results.json；新复现须指定复制好protocol.json的新输出目录。")
    doc_path = out/"辅助候选直接读出消融_实验结果.docx"
    doc.save(doc_path)
    (out/"实验结果说明.txt").write_text("\n".join(text), encoding="utf-8")
    print(doc_path)


if __name__ == "__main__":
    main()
