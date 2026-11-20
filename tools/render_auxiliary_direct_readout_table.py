#!/usr/bin/env python3
"""Render the verified direct-readout replacement ablation as a paper table."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/scannet_auxiliary_direct_readout_20261102/results.json"
DESTINATION = ROOT / "figure_assets/auxiliary_direct_readout_ablation_20261102"
ROWS = (
    ("current_p_all", "Direct proposals", "当前proposal直接输出"),
    ("current_both_all", "Direct proposals + anchors", "当前proposal＋anchor直接输出"),
    ("replace_plr_current12", "PLR replaced by direct proposals", "PLR替换为当前proposal Top-12"),
    ("replace_calr_current_all", "CALR replaced by direct anchors", "CALR替换为当前anchor直接输出"),
    ("full", "Full ReCaR-3D", "完整ReCaR-3D"),
)
CAPTION_ZH = (
    "表X　ScanNet-100上的恢复读出替代消融。各路线共享同一冻结WeDetect-Uni/Boxer候选流、"
    "1,787个终态原生框及相同HCSR分数。前两行只输出当前关键帧的全部有效候选；"
    "第三行以当前proposal Top-12替代PLR，保留CALR；第四行以当前全部anchor替代CALR，保留PLR。"
    "proposal与anchor沿用原系统的分支评分，不新增三维NMS或去重。"
    "报告同缓存、确定性稳定排序的类别无关AP；实际输出数量不强制相等。"
    "该表检验当前帧直接读出能否替代历史恢复，不替代与持久简单记忆的比较。"
)
CAPTION_EN = (
    "Recovery-readout replacement ablation on ScanNet-100. All variants share the frozen "
    "WeDetect-Uni/Boxer candidate stream, 1,787 terminal native boxes, and identical HCSR scores. "
    "Direct readout uses only candidates from the current keyframe. In the PLR replacement, "
    "the top 12 current proposals replace PLR while CALR is retained; in the CALR replacement, "
    "all current anchors replace CALR while PLR is retained. Branch-specific scoring is unchanged. "
    "AP uses deterministic stable score sorting; actual output counts are not forced to match. "
    "These are current-readout controls, not persistent-memory baselines."
)


def render(values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 16,
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(8.7, 2.65))
    fig.subplots_adjust(left=.015, right=.985, top=.97, bottom=.035)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top, middle, bottom = .96, .805, .045
    ax.plot([.01, .99], [top, top], color="black", lw=1.65)
    ax.plot([.01, .99], [middle, middle], color="black", lw=1.45)
    ax.plot([.01, .99], [bottom, bottom], color="black", lw=1.65)
    ax.plot([.595, .595], [bottom, top], color="black", lw=.7)
    xs = (.672, .802, .932)
    ax.text(.03, .884, "Recovery setting", va="center", fontsize=16)
    for x, threshold in zip(xs, (15, 25, 50)):
        ax.text(x, .884, rf"$AP_{{{threshold}}}\uparrow$", ha="center", va="center", fontsize=18)
    for index, ((arm, english, _), scores) in enumerate(zip(ROWS, values)):
        y = middle - (index+.5)*(middle-bottom)/len(ROWS)
        bold = arm == "full"
        ax.text(.03, y, english, va="center", fontsize=16,
                fontweight="bold" if bold else "normal")
        for x, score in zip(xs, scores):
            ax.text(x, y, f"{score:.2f}", ha="center", va="center", fontsize=18,
                    fontweight="bold" if bold else "normal")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(DESTINATION/f"recovery_readout_ablation.{suffix}", dpi=400,
                    facecolor="white", metadata={"Creator": "Verified ReCaR-3D result renderer"})
    plt.close(fig)


def word(values):
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    doc = Document()
    doc.sections[0].left_margin = doc.sections[0].right_margin = Cm(2)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(10)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    doc.add_paragraph("恢复读出替代消融（ScanNet-100）")
    table = doc.add_table(rows=1, cols=4)
    for cell, value in zip(table.rows[0].cells, ("恢复设置", "AP15 ↑", "AP25 ↑", "AP50 ↑")):
        cell.text = value
    for (arm, _, chinese), scores in zip(ROWS, values):
        cells = table.add_row().cells
        for column, value in enumerate((chinese, *[f"{s:.2f}" for s in scores])):
            cells[column].text = value
            if column:
                cells[column].paragraphs[0].alignment = 1
            if arm == "full":
                for run in cells[column].paragraphs[0].runs:
                    run.bold = True
    table.autofit = False
    for row in table.rows:
        for cell, width in zip(row.cells, (10.0, 1.9, 1.9, 1.9)):
            cell.width = Cm(width)
            borders = OxmlElement("w:tcBorders")
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                item = OxmlElement("w:"+edge)
                item.set(qn("w:val"), "nil")
                borders.append(item)
            cell._tc.get_or_add_tcPr().append(borders)
    for row_index, edge, size in ((0, "top", "12"), (0, "bottom", "8"), (len(ROWS), "bottom", "12")):
        for cell in table.rows[row_index].cells:
            borders = cell._tc.get_or_add_tcPr().find(qn("w:tcBorders"))
            item = borders.find(qn("w:"+edge))
            item.set(qn("w:val"), "single")
            item.set(qn("w:sz"), size)
            item.set(qn("w:color"), "000000")
    doc.add_paragraph(CAPTION_ZH)
    doc.save(DESTINATION/"恢复读出替代消融_可编辑表格.docx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx-only", action="store_true")
    args = parser.parse_args()
    result = json.loads(SOURCE.read_text())
    assert result["validation"]["status"] == "PASS" and result["native_count"] == 1787
    points = result["prefix_results"]["1.0"]
    values = [[points[arm]["stable_ap"][str(t)] for t in (.15, .25, .5)] for arm, _, _ in ROWS]
    assert all(values[-1][j] == max(v[j] for v in values) for j in range(3))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if args.docx_only:
        word(values)
        return
    render(values)
    with (DESTINATION/"table_values.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "setting", "AP15", "AP25", "AP50"])
        writer.writerows([[arm, english, *scores] for (arm, english, _), scores in zip(ROWS, values)])
    latex = ["% Requires \\usepackage{booktabs}", "\\begin{table}[t]", "\\centering",
             "\\setlength{\\tabcolsep}{4pt}", "\\small", "\\begin{tabular}{lccc}", "\\toprule",
             r"Recovery setting & $AP_{15}\uparrow$ & $AP_{25}\uparrow$ & $AP_{50}\uparrow$ \\", "\\midrule"]
    for (arm, english, _), scores in zip(ROWS, values):
        labels = [english, *[f"{v:.2f}" for v in scores]]
        if arm == "full":
            labels = ["\\textbf{"+v+"}" for v in labels]
        latex.append(" & ".join(labels)+r" \\")
    latex += ["\\bottomrule", "\\end{tabular}", "\\caption{"+CAPTION_EN+"}",
              "\\label{tab:recovery-direct-readout}", "\\end{table}"]
    (DESTINATION/"recovery_readout_ablation.tex").write_text("\n".join(latex)+"\n")
    (DESTINATION/"图表说明.txt").write_text(CAPTION_ZH+"\n\n"+CAPTION_EN+"\n", encoding="utf-8")
    manifest = {"source": str(SOURCE), "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "metric": "same-cache deterministic stable class-agnostic AP", "threshold_order": [.15, .25, .5],
        "arms": [row[0] for row in ROWS], "native_count": 1787, "model_rerun": False,
        "generation": "code-rendered table, not generative imagery", "actual_output_counts_matched": False}
    (DESTINATION/"table_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(DESTINATION)


if __name__ == "__main__":
    main()
