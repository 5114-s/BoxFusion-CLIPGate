#!/usr/bin/env python3
"""Render the finalized ReCaR-3D Markdown supplement to a readable DOCX."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


def plain(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", value)
    return value.replace("\\(", "").replace("\\)", "")


def set_cell_shading(cell, fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shade = OxmlElement("w:shd")
    shade.set(qn("w:fill"), fill)
    props.append(shade)


def set_run_font(run, latin: str, east_asia: str, size: float, bold=False) -> None:
    run.font.name = latin
    run.font.size = Pt(size)
    run.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)


def add_table(document: Document, lines: list[str]) -> None:
    rows = [[plain(cell) for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) >= 2 and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in rows[1]):
        rows.pop(1)
    if not rows:
        return
    width = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=width)
    table.style = "Table Grid"
    table.autofit = True
    for r_index, row in enumerate(rows):
        for c_index in range(width):
            value = row[c_index] if c_index < len(row) else ""
            cell = table.cell(r_index, c_index)
            cell.text = value
            if r_index == 0:
                set_cell_shading(cell, "D9EAF7")
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                for run in paragraph.runs:
                    set_run_font(run, "Times New Roman", "宋体", 7.5, r_index == 0)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def build(markdown: Path, output: Path,
          footer_text: str = "ReCaR-3D Supplementary Material — 2026-10-12") -> None:
    lines = markdown.read_text(encoding="utf-8").splitlines()
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.55)
    section.right_margin = Inches(0.55)

    styles = document.styles
    for style_name, size, bold in (
        ("Normal", 9.0, False),
        ("Title", 18.0, True),
        ("Heading 1", 14.0, True),
        ("Heading 2", 11.5, True),
        ("Heading 3", 10.0, True),
    ):
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = bold
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体" if bold else "宋体")

    index = 0
    first_title = True
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("|"):
            block = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                block.append(lines[index])
                index += 1
            add_table(document, block)
            continue
        match = re.match(r"^(#{1,3})\s+(.*)$", line)
        if match:
            level = len(match.group(1))
            title = plain(match.group(2))
            if first_title and level == 1:
                paragraph = document.add_paragraph(style="Title")
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                first_title = False
            else:
                paragraph = document.add_paragraph(style=f"Heading {level}")
            paragraph.add_run(title)
            index += 1
            continue
        if line.startswith("> "):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.space_after = Pt(5)
            run = paragraph.add_run(plain(line[2:]))
            set_run_font(run, "Times New Roman", "宋体", 8.5)
            index += 1
            continue
        bullet = re.match(r"^[-*]\s+(.*)$", line)
        numbered = re.match(r"^\d+\.\s+(.*)$", line)
        if bullet or numbered:
            paragraph = document.add_paragraph(style="List Bullet" if bullet else "List Number")
            paragraph.add_run(plain((bullet or numbered).group(1)))
            index += 1
            continue
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(4)
        paragraph.paragraph_format.line_spacing = 1.08
        run = paragraph.add_run(plain(line))
        set_run_font(run, "Times New Roman", "宋体", 9.0)
        index += 1

    footer = document.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run(footer_text)
    set_run_font(run, "Times New Roman", "宋体", 8.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--footer", default="ReCaR-3D Supplementary Material — 2026-10-12")
    args = parser.parse_args()
    build(args.markdown, args.output, args.footer)


if __name__ == "__main__":
    main()
