"""Synchronize the course-paper DOCX with the audited Section 3 and Figure 1.

The source DOCX intentionally stores display formulae as centered text rather
than Office Math.  This updater preserves that convention while adding the
audited formulae, visible equation numbers, and a compact parameter table.
"""

from __future__ import annotations

from html import escape
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "course_paper_cross_modal_residual_mining.docx"
FIGURE1 = ROOT / "figures/course_paper/figure1_method_pipeline.png"


def _text(value: str) -> str:
    return escape(value, quote=False)


def body_paragraph(value: str) -> str:
    return (
        '<w:p><w:pPr><w:ind w:firstLine="420"/>'
        '<w:spacing w:before="0" w:after="120" w:line="360" '
        'w:lineRule="auto"/></w:pPr><w:r><w:t xml:space="preserve">'
        f"{_text(value)}</w:t></w:r></w:p>"
    )


def heading2(value: str) -> str:
    return (
        '<w:p><w:pPr><w:pStyle w:val="Heading2"/><w:jc w:val="left"/>'
        '<w:spacing w:before="120" w:after="80" w:line="360" '
        'w:lineRule="auto"/></w:pPr><w:r><w:t xml:space="preserve">'
        f"{_text(value)}</w:t></w:r></w:p>"
    )


def equation(lines: list[str], number: int) -> str:
    runs: list[str] = []
    for index, line in enumerate(lines):
        suffix = f"                                      ({number})" if index == len(lines) - 1 else ""
        runs.append(f'<w:r><w:t xml:space="preserve">{_text(line + suffix)}</w:t>')
        if index < len(lines) - 1:
            runs.append("<w:br/>")
        runs.append("</w:r>")
    return (
        '<w:p><w:pPr><w:jc w:val="center"/>'
        '<w:spacing w:before="0" w:after="60" w:line="300" '
        'w:lineRule="auto"/></w:pPr>'
        + "".join(runs)
        + "</w:p>"
    )


def caption(value: str) -> str:
    return (
        '<w:p><w:pPr><w:jc w:val="center"/>'
        '<w:spacing w:before="80" w:after="60" w:line="300" '
        'w:lineRule="auto"/></w:pPr><w:r><w:rPr><w:b/></w:rPr>'
        f'<w:t xml:space="preserve">{_text(value)}</w:t></w:r></w:p>'
    )


def table_cell(value: str, width: int, *, bold: bool = False, shade: bool = False) -> str:
    run_props = "<w:rPr><w:b/></w:rPr>" if bold else ""
    shading = '<w:shd w:val="clear" w:color="auto" w:fill="DCE6F1"/>' if shade else ""
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shading}</w:tcPr>'
        '<w:p><w:pPr><w:spacing w:before="0" w:after="30" w:line="260" '
        'w:lineRule="auto"/></w:pPr><w:r>'
        f'{run_props}<w:t xml:space="preserve">{_text(value)}</w:t></w:r></w:p></w:tc>'
    )


def parameter_table() -> str:
    rows = [
        ("Stage", "Parameter group", "Fixed value"),
        ("Residual generation", "Class-agnostic anchor matching", "maximum AABB IoU ≤ 0.15"),
        ("Stage 1 depth", "Selected views; sampling; ray margin", "Top-5 by area; stride 4; 0.05 m"),
        ("Stage 1 depth", "Valid depth; samples per view", "0.10–8.00 m; at least 16"),
        ("Stage 1 view rules", "Supportive; contradictory free space", "support ≥ 0.10, free ≤ 0.50; free > 0.50 and > support"),
        ("Stage 1 appearance", "Encoder; input; missing-pair prior", "DINOv3-S/16+; 960 × 960; 0.50"),
        ("Stage 1 output", "Per-scene verification budget", "Top-10 by r_C1"),
        ("Stage 2 projection", "Area; initial mask match", "≥ 25 px; IoU ≥ 0.02 or containment/coverage ≥ 0.10"),
        ("Stage 2 mask/depth", "Mask score; valid depth pixels", "≥ 0.50; ≥ 24 in 0.10–8.00 m"),
        ("Stage 2 local support", "Expansion; voxel; per-view component", "1.25×; 0.05 m; n_cc ≥ 16, f_in ≥ 0.20"),
        ("Stage 2 final gate", "Strong views; total points; mean support", "H ≥ 2; N_cc ≥ 64; mean f_exp ≥ 0.25"),
        ("Materialization", "Per-scene final rank budget", "rank ≤ 5 and g_md = 1"),
    ]
    widths = (1900, 3500, 3600)
    body: list[str] = []
    for row_index, row in enumerate(rows):
        cells = "".join(
            table_cell(value, width, bold=row_index == 0, shade=row_index == 0)
            for value, width in zip(row, widths)
        )
        body.append(f"<w:tr>{cells}</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="9000" w:type="dxa"/><w:tblLook w:val="04A0" '
        'w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
        'w:noHBand="0" w:noVBand="1"/></w:tblPr>'
        '<w:tblGrid><w:gridCol w:w="1900"/><w:gridCol w:w="3500"/>'
        '<w:gridCol w:w="3600"/></w:tblGrid>'
        + "".join(body)
        + "</w:tbl>"
    )


def section_three_block() -> str:
    parts = [
        heading2("3.3 Stage 1: geometry and appearance ranking"),
        body_paragraph(
            "For each residual proposal \\(p\\), the R2a observer selects up to five valid views with the largest projected box areas. At pixel stride four, every sampled depth ray is classified as box support (\\(S\\)), occluded (\\(O\\)), free-space contradiction (\\(F\\)), or invalid (\\(I\\)). Let \\(n_t^k\\) be the count of class \\(k\\) in view \\(t\\), \\(n_t=\\sum_k n_t^k\\), \\(v_t\\) indicate a valid projection, and \\(\\rho_t^k=n_t^k/n_t\\). Supportive and contradictory views are defined explicitly as"
        ),
        equation(
            [
                r"u_t(p)=\mathbb{1}[v_t=1 \land n_t\ge16 \land \rho_t^S\ge0.10 \land \rho_t^F\le0.50],",
                r"c_t(p)=\mathbb{1}[v_t=1 \land n_t\ge16 \land \rho_t^F>0.50 \land \rho_t^F>\rho_t^S].",
            ],
            4,
        ),
        body_paragraph(
            "Let \\(V=\\sum_t v_t\\), \\(U=\\sum_t u_t\\), and \\(C=\\sum_t c_t\\). The aggregate fractions \\(\\rho^S,\\rho^F,\\rho^I\\) are computed after pooling the corresponding ray counts across selected views. The three depth reliability factors are"
        ),
        equation(
            [
                r"q_{con}(p)=(U+1)/(V+2), \quad q_{ctr}(p)=C/V \; (V>0;\ 0\ otherwise),",
                r"q_{dep}(p)=\rho^S/(\rho^S+\rho^F)\,(1-\rho^I).",
            ],
            5,
        ),
        body_paragraph(
            "A zero denominator in \\(q_{dep}\\) returns zero. This smoothed formulation rewards repeated support, penalizes observed free space, and remains well defined for sparsely observed proposals."
        ),
        body_paragraph(
            "For appearance consistency, support rays are projected into the RGB image and mapped to unique cells of a frozen DINOv3-S/16+ dense feature map. The selected cell features are averaged and \\(\\ell_2\\)-normalized to form \\(z_t(p)\\). If \\(m\\) valid feature views are available, CMRM uses the pairwise cosine mean, not the median:"
        ),
        equation(
            [
                r"\bar c(p)=2/[m(m-1)]\sum_{t<u}z_t(p)^\top z_u(p),",
                r"q_{app}(p)=\operatorname{clip}((\bar c(p)+1)/2,0,1)\ (m\ge2);\quad q_{app}=0.5\ (m<2).",
            ],
            6,
        ),
        body_paragraph(
            "Let \\(s_d(p)\\) be the frozen TR3D confidence. Equations (5) and (6) enter the exact deterministic ranking score"
        ),
        equation(
            [
                r"r_{depth}(p)=s_d(p)(0.5+0.5q_{con})(1-0.5q_{ctr})(0.5+0.5q_{dep}),",
                r"r_{C1}(p)=r_{depth}(p)(0.75+0.25q_{app}).",
            ],
            7,
        ),
        body_paragraph(
            "There is no candidate-pool normalization or learned coefficient in Eq. (7). Candidates are sorted stably by \\(r_{C1}\\) within each scene, and the Top-10 are forwarded to Stage 2, leaving 1,000 source candidates. Temporal span and center/size stability are logged for audit but are not used by the reported route. In oracle analysis, this pool contains 209, 194, and 106 novel matches at IoU 0.15, 0.25, and 0.50."
        ),
        heading2("3.4 Stage 2: mask and depth verification"),
        body_paragraph(
            "Each Top-10 candidate is projected into scheduled RGB frames and matched class agnostically against precomputed promptable-mask proposals; the reported run does not issue a new rectangular prompt for every residual. For projected box footprint \\(B_t\\) and mask \\(M_t\\), define mask–box IoU \\(J_t\\), mask containment \\(C_t^M\\), and box coverage \\(C_t^B\\). A mask is eligible when"
        ),
        equation(
            [
                r"J_t=|M_t\cap B_t|/|M_t\cup B_t|,\quad C_t^M=|M_t\cap B_t|/|M_t|,\quad C_t^B=|M_t\cap B_t|/|B_t|,",
                r"M_t\ eligible \Longleftrightarrow J_t\ge0.02\ \lor\ (C_t^M\ge0.10\ \land\ C_t^B\ge0.10).",
            ],
            8,
        ),
        body_paragraph("Among eligible masks, the deterministic evidence score selects one mask per view:"),
        equation(
            [
                r"E_t=0.15(s_t^M+J_t+C_t^M+C_t^B+f_t^{exp})+0.10f_t^{in}",
                r"\quad +0.05[f_t^{cc}+\min(n_t^D/24,1)+\min(n_t^{cc}/16,1)].",
            ],
            9,
        ),
        body_paragraph(
            "Here \\(s_t^M\\) is mask confidence, \\(n_t^D\\) is the valid mask-depth pixel count, and \\(f_t^{exp}\\) is the fraction of backprojected points inside a 1.25-times expanded 3D box. Points in that box are voxelized at 0.05 m with 26-neighbor connectivity. The best box-aligned component has \\(n_t^{cc}\\) points, original-box fraction \\(f_t^{in}\\), and expanded-support fraction \\(f_t^{cc}\\). A projection provides strong support only when"
        ),
        equation(
            [
                r"h_t(p)=\mathbb{1}[A_t\ge25 \land s_t^M\ge0.50 \land C_t^M\ge0.10 \land C_t^B\ge0.10",
                r"\quad \land n_t^D\ge24 \land f_t^{exp}\ge0.15 \land n_t^{cc}\ge16 \land f_t^{in}\ge0.20].",
            ],
            10,
        ),
        body_paragraph(
            "where \\(A_t\\) is projected area in pixels. Let \\(H=\\sum_t h_t\\), \\(N_{cc}=\\sum_t h_tn_t^{cc}\\), and let \\(\\bar f^{exp}\\) be the mean expanded-box support over strong views. The complete mask-depth gate is"
        ),
        equation(
            [
                r"\bar f^{exp}(p)=H^{-1}\sum_t h_t f_t^{exp}\ (H>0;\ 0\ otherwise),",
                r"g_{md}(p)=\mathbb{1}[H\ge2 \land N_{cc}\ge64 \land \bar f^{exp}\ge0.25].",
            ],
            11,
        ),
        body_paragraph(
            "Equation (11) is stricter than a two-view count alone: it also requires 64 component points across strong views and mean expanded-box support of at least 0.25. It is an association rule across transactions: “mask agreement AND metric-depth structure in at least two views implies candidate validity.”"
        ),
        body_paragraph(
            "The initial mask-plus-depth route accepts 294 candidates over all 100 scenes. Although it is much cleaner than the 1,000-candidate pool, held-out precision remains slightly below the pre-registered targets. We therefore report it transparently and analyze a stricter rule rather than presenting it as a successful final classifier."
        ),
        caption("Table 1. Fixed inference parameters for the reported CMRM route. Diagnostic-only gates are excluded."),
        parameter_table(),
    ]
    return "".join(parts)


def paragraph_bounds(xml: str, needle: str) -> tuple[int, int]:
    position = xml.find(needle)
    if position < 0:
        raise ValueError(f"paragraph text not found: {needle}")
    start = xml.rfind("<w:p>", 0, position)
    end = xml.find("</w:p>", position)
    if start < 0 or end < 0:
        raise ValueError(f"paragraph boundary not found: {needle}")
    return start, end + len("</w:p>")


def repair_legacy_boundaries(xml: str) -> str:
    """Repair documents written by the first updater revision.

    The first revision searched for ``<w:p`` and could therefore stop at a
    nested ``<w:pStyle>`` element.  These three signatures are exact and
    occur only around the two replaced blocks.
    """

    fixes = (
        (
            '<w:p><w:pPr><w:p><w:pPr><w:pStyle w:val="Heading2"/>',
            '<w:p><w:pPr><w:pStyle w:val="Heading2"/>',
        ),
        (
            '</w:tbl><w:pStyle w:val="Heading2"/>',
            '</w:tbl><w:p><w:pPr><w:pStyle w:val="Heading2"/>',
        ),
        (
            '</w:p><w:p><w:p><w:pPr><w:ind w:firstLine="420"/>',
            '</w:p><w:p><w:pPr><w:ind w:firstLine="420"/>',
        ),
    )
    for old, new in fixes:
        if old in xml:
            xml = xml.replace(old, new, 1)
    return xml


def synchronize_updated_document(xml: str) -> str:
    replacements = (
        (
            r"Let \(V=\sum_t v_t\), \(U=\sum_t u_t\), and \(C=\sum_t c_t\).",
            r"Equation (4) fixes the per-view decisions without learned thresholds. Let \(V=\sum_t v_t\), \(U=\sum_t u_t\), and \(C=\sum_t c_t\).",
        ),
        (
            "Among eligible masks, the deterministic evidence score selects one mask per view:",
            "Among masks admitted by Eq. (8), the deterministic evidence score selects one mask per view:",
        ),
        (
            r"Here \(s_t^M\) is mask confidence,",
            r"The eligible mask with maximum Eq. (9) is retained. Here \(s_t^M\) is mask confidence,",
        ),
        (
            r"where \(A_t\) is projected area in pixels.",
            r"In Eq. (10), \(A_t\) is projected area in pixels.",
        ),
    )
    for old, new in replacements:
        if old in xml:
            xml = xml.replace(old, new, 1)
    return xml


def replace_once(xml: str, old: str, new: str) -> str:
    count = xml.count(old)
    if count != 1:
        raise ValueError(f"expected one match, found {count}: {old[:100]}")
    return xml.replace(old, new, 1)


def update_document(xml: str) -> str:
    xml = xml.replace("DINOv2", "DINOv3")
    xml = replace_once(xml, "using promptable image masks", "using cached promptable masks")
    xml = replace_once(
        xml,
        "It then uses promptable segmentation masks inspired by the Segment Anything framework [8] and real registered depth to confirm whether the projected 3D box is supported in multiple images.",
        "It then matches projected hypotheses to cached promptable segmentation masks inspired by the Segment Anything framework [8] and uses registered depth to test whether a 3D box is supported in multiple images.",
    )
    xml = replace_once(
        xml,
        "DINOv3 [7] provides self-supervised visual features that generalize across image-level and pixel-level tasks. Compared with a category score, such features can express whether crops from different viewpoints depict the same physical object even when the category is unknown. CMRM uses DINOv3 as a frozen feature extractor. No ScanNet validation labels are used to fine-tune it.",
        "DINOv3 [7] provides high-quality self-supervised dense visual features. Compared with a category score, such features can express whether depth-supported regions from different viewpoints depict the same physical object even when the category is unknown. CMRM uses a frozen DINOv3-S/16+ encoder; no ScanNet validation labels are used to fine-tune it.",
    )

    xml = replace_once(
        xml,
        r"\mathcal{R}=\{p\in\mathcal{P}: \max_{a\in\mathcal{A}} \operatorname{IoU}_{3D}(p,a)&lt;\tau_m\}.          (3)",
        r"\mathcal{R}=\{p\in\mathcal{P}: \max_{a\in\mathcal{A}} \operatorname{IoU}_{\mathrm{AABB}}(p,a)\le 0.15\}.          (3)",
    )
    xml = replace_once(
        xml,
        "Equations (1)–(3) define the streaming observations, frozen detections, and unmatched residual pool, respectively. The goal is to learn or construct a selector",
        "Equations (1)–(3) define the streaming observations, frozen detections, and unmatched residual pool, respectively. The residual test is class agnostic and uses axis-aligned 3D IoU in the common world frame. The goal is to construct a selector",
    )
    xml = replace_once(
        xml,
        "TR3D is used as a complementary proposal source because its sparse 3D convolutional representation differs from frame-level box regression. Its predictions are transformed to the same world frame. Any candidate already explained by an anchor box is removed.",
        "TR3D is used as a complementary proposal source because its sparse 3D convolutional representation differs from frame-level box regression. Its predictions are transformed to the same world frame. Equation (3) removes every proposal whose maximum class-agnostic AABB IoU with the frozen anchor exceeds 0.15.",
    )

    stage_start, _ = paragraph_bounds(xml, "3.3 Stage 1: geometry and appearance ranking")
    stage_end, _ = paragraph_bounds(xml, "3.5 Conservative intersection and score-safe materialization")
    xml = xml[:stage_start] + section_three_block() + xml[stage_end:]

    formula_updates = [
        (
            r"g(p)=g_{md}(p)\land \mathbb{1}[\operatorname{rank}_{scene}(p)\le 5].                    (7)",
            r"g(p)=g_{md}(p)\land \mathbb{1}[\operatorname{rank}_{scene}(p)\le 5].                   (12)",
        ),
        ("The gate in Eq. (7) accepts 170 candidates.", "The gate in Eq. (12) accepts 170 candidates."),
        (
            r"0&lt;s(p)&lt;s_{min}\quad \forall p\in\mathcal{C}.                                      (8)",
            r"0&lt;s(p)&lt;s_{min}\quad \forall p\in\mathcal{C}.                                     (13)",
        ),
        ("Therefore, by Eq. (8),", "Therefore, by Eq. (13),"),
        (
            r"P(hit@\tau)=\frac{|\{p:\max_j IoU_{3D}(p,b_j^{gt})\ge\tau\}|}{|\mathcal{C}|}.          (9)",
            r"P(hit@\tau)=\frac{|\{p:\max_j IoU_{3D}(p,b_j^{gt})\ge\tau\}|}{|\mathcal{C}|}.         (14)",
        ),
        ("Equation (9) measures independent candidate hit precision.", "Equation (14) measures independent candidate hit precision."),
    ]
    for old, new in formula_updates:
        xml = replace_once(xml, old, new)

    table_updates = [
        ("Table 1 reproduces the relevant online comparison", "Table 2 reproduces the relevant online comparison"),
        ("Table 1. Results reported by the original BoxFusion paper", "Table 2. Results reported by the original BoxFusion paper"),
        ("Table 2 shows how progressively stronger rules", "Table 3 shows how progressively stronger rules"),
        ("Table 2. Residual candidate quality", "Table 3. Residual candidate quality"),
        ("Table 3 compares the frozen R3-active anchor", "Table 4 compares the frozen R3-active anchor"),
        ("Table 3. Paired ScanNet-100 detection results", "Table 4. Paired ScanNet-100 detection results"),
    ]
    for old, new in table_updates:
        xml = replace_once(xml, old, new)

    xml = replace_once(xml, "DINOv3 cross-view appearance, promptable masks", "DINOv3 cross-view appearance, cached promptable masks")

    ref_start, ref_end = paragraph_bounds(xml, "[7] M. Oquab et al.")
    xml = (
        xml[:ref_start]
        + body_paragraph('[7] O. Siméoni et al., “DINOv3,” arXiv preprint arXiv:2508.10104, 2025.')
        + xml[ref_end:]
    )

    if "DINOv2" in xml:
        raise ValueError("stale DINOv2 text remains")
    for number in range(1, 15):
        if f"({number})" not in xml:
            raise ValueError(f"equation number ({number}) missing")
    return xml


def main() -> None:
    if not DOCX.is_file() or not FIGURE1.is_file():
        raise FileNotFoundError("DOCX or Figure 1 is missing")
    target = DOCX.with_suffix(".docx.tmp")
    with ZipFile(DOCX, "r") as source, ZipFile(target, "w", ZIP_DEFLATED) as output:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename == "word/document.xml":
                document = repair_legacy_boundaries(payload.decode("utf-8"))
                if "For every residual proposal" in document:
                    document = update_document(document)
                document = synchronize_updated_document(document)
                ET.fromstring(document)
                payload = document.encode("utf-8")
            elif info.filename == "word/media/image1.png":
                payload = FIGURE1.read_bytes()
            output.writestr(info, payload)
    os.replace(target, DOCX)
    print(DOCX)


if __name__ == "__main__":
    main()
