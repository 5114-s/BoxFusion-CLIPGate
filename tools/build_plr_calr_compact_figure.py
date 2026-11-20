#!/usr/bin/env python3
"""Draw a compact, code-faithful PLR/CALR method panel."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figure_assets/plr_calr_compact_20261103"
W, H = 3000, 1120

INK = "#101820"
MUTED = "#596879"
ARROW = "#758292"
PLR = "#ff5a1f"
PLR_LIGHT = "#fff1ea"
CALR = "#c88700"
CALR_LIGHT = "#fff6da"
NATIVE = "#1769e8"
GREEN = "#18a957"


def face(size: int, bold: bool = False, italic: bool = False):
    if bold:
        name = "DejaVuSans-Bold.ttf"
    elif italic:
        name = "DejaVuSans-Oblique.ttf"
    else:
        name = "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def center(draw, xy, text, font, fill=INK):
    b = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (b[2] - b[0]) / 2, xy[1] - (b[3] - b[1]) / 2),
              text, font=font, fill=fill)


def arrow(draw, p1, p2, color=ARROW, width=11):
    x1, y1 = p1
    x2, y2 = p2
    draw.line((x1, y1, x2 - 25, y2), fill=color, width=width)
    draw.polygon(((x2 - 28, y2 - 20), (x2, y2), (x2 - 28, y2 + 20)), fill=color)


def dashed_line(draw, p1, p2, color, width=8, dash=15, gap=12):
    x1, y1 = p1
    x2, y2 = p2
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length == 0:
        return
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    pos = 0.0
    while pos < length:
        end = min(length, pos + dash)
        draw.line((x1 + ux * pos, y1 + uy * pos,
                   x1 + ux * end, y1 + uy * end), fill=color, width=width)
        pos += dash + gap


def cuboid(draw, x, y, w, h, d, color, width=7, fill=None, alpha=None):
    # x,y is front-top-left.  Positive depth goes right/up.
    dx, dy = d, -int(d * 0.58)
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y), (x + dx, y + dy), (x + w + dx, y + dy), (x + w, y)]
    side = [(x + w, y), (x + w + dx, y + dy),
            (x + w + dx, y + h + dy), (x + w, y + h)]
    if fill:
        draw.polygon(front, fill=fill)
        draw.polygon(top, fill=fill)
        draw.polygon(side, fill=fill)
    for poly in (front, top, side):
        draw.line(poly + [poly[0]], fill=color, width=width, joint="curve")


def star_points(cx, cy, outer=28, inner=12, n=5):
    import math
    pts = []
    for i in range(2 * n):
        r = outer if i % 2 == 0 else inner
        a = -math.pi / 2 + i * math.pi / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def voxel(draw, x, y, size=210, cells=4):
    # Front grid plus top/right faces gives a compact world-space voxel icon.
    d = 62
    cuboid(draw, x, y, size, size, d, "#7893b5", width=5, fill="#f8fbff")
    for i in range(1, cells):
        xx = x + size * i / cells
        yy = y + size * i / cells
        draw.line((xx, y, xx, y + size), fill="#8ca4c1", width=4)
        draw.line((x, yy, x + size, yy), fill="#8ca4c1", width=4)
    # Indicate depth subdivisions on top face.
    for i in range(1, cells):
        t = i / cells
        draw.line((x + d * t, y - d * .58 * t,
                   x + size + d * t, y - d * .58 * t), fill="#8ca4c1", width=4)


def build_png(path: Path):
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((18, 18, W - 18, 530), radius=45, fill="#fffdfb",
                        outline=PLR, width=6)
    d.line((40, 560, W - 40, 560), fill="#c8cdd3", width=5)
    d.rounded_rectangle((18, 590, W - 18, H - 18), radius=45, fill="#fffdf8",
                        outline=CALR, width=6)

    d.text((55, 45), "PLR | Proposal-Level Recovery", font=face(58, True), fill=INK)
    d.text((55, 615), "CALR | Causal Anchor-Level Recovery", font=face(58, True), fill=INK)

    # PLR: track and verify.
    center(d, (400, 150), "Track + 3-Frame Verify", face(40, True))
    xs = [230, 390, 550]
    for i, x in enumerate(xs, 1):
        center(d, (x + 45, 215), f"t{i}", face(32, italic=True))
        cuboid(d, x, 250, 70, 65, 28, PLR, width=5, fill=PLR_LIGHT)
        d.ellipse((x + 28, 365, x + 46, 383), fill=INK)
    d.line((205, 374, 660, 374), fill=INK, width=7)
    d.polygon(((660, 374), (632, 354), (632, 394)), fill=INK)
    arrow(d, (695, 305), (820, 305))

    # PLR: medoid.
    center(d, (1050, 150), "AABB-IoU Medoid", face(40, True))
    cuboid(d, 875, 265, 115, 95, 38, "#ffb296", width=4)
    cuboid(d, 925, 245, 115, 95, 38, "#ffceb9", width=4)
    cuboid(d, 1010, 275, 125, 100, 42, PLR, width=7, fill=PLR_LIGHT)
    center(d, (1070, 420), "selected observation", face(28), MUTED)
    arrow(d, (1220, 305), (1370, 305))

    # PLR: native check + ranked dedup, with the map used as a reference only.
    center(d, (1760, 150), "Native Check + Ranked Dedup.", face(40, True))
    center(d, (1885, 205), "Native Map N⁽ᵗ⁾", face(30, True), NATIVE)
    dashed_line(d, (1885, 228), (1885, 265), NATIVE, width=7, dash=10, gap=8)
    cuboid(d, 1445, 295, 125, 100, 42, PLR, width=7, fill=PLR_LIGHT)
    center(d, (1510, 430), "medoid", face(27), MUTED)
    center(d, (1650, 340), "IoU", face(34, True), MUTED)
    cuboid(d, 1815, 295, 125, 100, 42, NATIVE, width=7, fill="#edf5ff")
    center(d, (1880, 430), "map reference", face(27), MUTED)
    d.line((1580, 340, 1625, 340), fill=ARROW, width=7)
    d.line((1690, 340, 1800, 340), fill=ARROW, width=7)
    center(d, (2050, 340), "✓", face(78, True), GREEN)
    arrow(d, (2130, 335), (2380, 335), color=PLR, width=12)
    center(d, (2690, 285), "PLR", face(53, True), PLR)
    center(d, (2690, 350), "Outputs", face(53, True), PLR)

    # CALR: voxel consensus.
    center(d, (560, 715), "3-Frame Voxel Consensus", face(40, True))
    voxel(d, 410, 790, 220, 4)
    dots = [(475, 875, NATIVE, "t₁"), (530, 935, "#eeb400", "t₂"),
            (575, 900, PLR, "t₃")]
    for x, y, c, t in dots:
        d.ellipse((x - 16, y - 16, x + 16, y + 16), fill=c)
        d.text((x + 20, y - 24), t, font=face(30), fill=INK)
    arrow(d, (760, 900), (980, 900))

    # CALR: exemplar.
    center(d, (1370, 715), "Highest-Score Exemplar", face(40, True))
    cuboid(d, 1190, 830, 250, 140, 75, CALR, width=8, fill=CALR_LIGHT)
    d.polygon(star_points(1315, 900, 32, 14), fill="#edaa00")
    center(d, (1315, 1025), "highest raw score", face(28, italic=True), MUTED)

    # Keep score mapping as a compact arrow annotation rather than a module.
    arrow(d, (1555, 900), (2505, 900), color=CALR, width=12)
    center(d, (2030, 955), "tail-score ranked", face(29), MUTED)
    center(d, (2735, 850), "CALR", face(53, True), CALR)
    center(d, (2735, 915), "Outputs", face(53, True), CALR)

    im.save(path, dpi=(600, 600), quality=96)


def svg_cuboid(x, y, w, h, dep, color, fill="none", width=7, opacity=1.0):
    dx, dy = dep, -dep * .58
    front = f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}"
    top = f"{x},{y} {x+dx},{y+dy} {x+w+dx},{y+dy} {x+w},{y}"
    side = f"{x+w},{y} {x+w+dx},{y+dy} {x+w+dx},{y+h+dy} {x+w},{y+h}"
    return "".join(
        f'<polygon points="{pts}" fill="{fill}" stroke="{color}" stroke-width="{width}" opacity="{opacity}" stroke-linejoin="round"/>'
        for pts in (front, top, side)
    )


def build_svg(path: Path):
    # The SVG mirrors the PNG but keeps all text and diagram primitives vector.
    s = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
    <defs>
      <marker id="a" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{ARROW}"/></marker>
      <marker id="ap" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{PLR}"/></marker>
      <marker id="ac" markerWidth="16" markerHeight="16" refX="14" refY="7" orient="auto"><path d="M0,0 L14,7 L0,14 Z" fill="{CALR}"/></marker>
      <style>
      .t {{font-family:Arial,Helvetica,sans-serif; fill:{INK}}}
      .h {{font-size:58px;font-weight:700}} .sh {{font-size:40px;font-weight:700}}
      .sm {{font-size:28px;fill:{MUTED}}} .mid {{font-size:34px;font-weight:700}}
      </style>
    </defs>
    <rect x="18" y="18" width="2964" height="512" rx="45" fill="#fffdfb" stroke="{PLR}" stroke-width="6"/>
    <line x1="40" y1="560" x2="2960" y2="560" stroke="#c8cdd3" stroke-width="5"/>
    <rect x="18" y="590" width="2964" height="512" rx="45" fill="#fffdf8" stroke="{CALR}" stroke-width="6"/>
    <text x="55" y="91" class="t h">PLR | Proposal-Level Recovery</text>
    <text x="55" y="661" class="t h">CALR | Causal Anchor-Level Recovery</text>
    <text x="400" y="162" text-anchor="middle" class="t sh">Track + 3-Frame Verify</text>
    ''']
    for i, x in enumerate((230, 390, 550), 1):
        s.append(f'<text x="{x+45}" y="225" text-anchor="middle" class="t mid" font-style="italic">t{i}</text>')
        s.append(svg_cuboid(x, 250, 70, 65, 28, PLR, PLR_LIGHT, 5))
        s.append(f'<circle cx="{x+37}" cy="374" r="9" fill="{INK}"/>')
    s += [
        f'<line x1="205" y1="374" x2="660" y2="374" stroke="{INK}" stroke-width="7" marker-end="url(#a)"/>',
        f'<line x1="695" y1="305" x2="820" y2="305" stroke="{ARROW}" stroke-width="11" marker-end="url(#a)"/>',
        '<text x="1050" y="162" text-anchor="middle" class="t sh">AABB-IoU Medoid</text>',
        svg_cuboid(875, 265, 115, 95, 38, "#ffb296", "none", 4, .8),
        svg_cuboid(925, 245, 115, 95, 38, "#ffceb9", "none", 4, .8),
        svg_cuboid(1010, 275, 125, 100, 42, PLR, PLR_LIGHT, 7),
        f'<text x="1070" y="430" text-anchor="middle" class="t sm">selected observation</text>',
        f'<line x1="1220" y1="305" x2="1370" y2="305" stroke="{ARROW}" stroke-width="11" marker-end="url(#a)"/>',
        '<text x="1760" y="162" text-anchor="middle" class="t sh">Native Check + Ranked Dedup.</text>',
        f'<text x="1885" y="215" text-anchor="middle" class="t mid" fill="{NATIVE}">Native Map N⁽ᵗ⁾</text>',
        f'<line x1="1885" y1="228" x2="1885" y2="265" stroke="{NATIVE}" stroke-width="7" stroke-dasharray="10 8"/>',
        svg_cuboid(1445, 295, 125, 100, 42, PLR, PLR_LIGHT, 7),
        svg_cuboid(1815, 295, 125, 100, 42, NATIVE, "#edf5ff", 7),
        f'<text x="1510" y="440" text-anchor="middle" class="t sm">medoid</text>',
        f'<text x="1650" y="351" text-anchor="middle" class="t mid" fill="{MUTED}">IoU</text>',
        f'<text x="1880" y="440" text-anchor="middle" class="t sm">map reference</text>',
        f'<line x1="1580" y1="340" x2="1625" y2="340" stroke="{ARROW}" stroke-width="7"/>',
        f'<line x1="1690" y1="340" x2="1800" y2="340" stroke="{ARROW}" stroke-width="7"/>',
        f'<text x="2050" y="365" text-anchor="middle" font-family="Arial" font-size="78" font-weight="700" fill="{GREEN}">✓</text>',
        f'<line x1="2130" y1="335" x2="2380" y2="335" stroke="{PLR}" stroke-width="12" marker-end="url(#ap)"/>',
        f'<text x="2690" y="300" text-anchor="middle" class="t h" fill="{PLR}">PLR</text>',
        f'<text x="2690" y="365" text-anchor="middle" class="t h" fill="{PLR}">Outputs</text>',
        '<text x="560" y="727" text-anchor="middle" class="t sh">3-Frame Voxel Consensus</text>',
        svg_cuboid(410, 790, 220, 220, 62, "#7893b5", "#f8fbff", 5),
    ]
    # Compact front-face voxel grid.
    for i in range(1, 4):
        x = 410 + 220 * i / 4
        y = 790 + 220 * i / 4
        s.append(f'<line x1="{x}" y1="790" x2="{x}" y2="1010" stroke="#8ca4c1" stroke-width="4"/>')
        s.append(f'<line x1="410" y1="{y}" x2="630" y2="{y}" stroke="#8ca4c1" stroke-width="4"/>')
    for x, y, color, label in ((475, 875, NATIVE, "t₁"), (530, 935, "#eeb400", "t₂"), (575, 900, PLR, "t₃")):
        s.append(f'<circle cx="{x}" cy="{y}" r="16" fill="{color}"/><text x="{x+22}" y="{y+8}" class="t mid">{label}</text>')
    s += [
        f'<line x1="760" y1="900" x2="980" y2="900" stroke="{ARROW}" stroke-width="11" marker-end="url(#a)"/>',
        '<text x="1370" y="727" text-anchor="middle" class="t sh">Highest-Score Exemplar</text>',
        svg_cuboid(1190, 830, 250, 140, 75, CALR, CALR_LIGHT, 8),
        f'<text x="1315" y="922" text-anchor="middle" font-family="Arial" font-size="64" fill="#edaa00">★</text>',
        f'<text x="1315" y="1035" text-anchor="middle" class="t sm" font-style="italic">highest raw score</text>',
        f'<line x1="1555" y1="900" x2="2505" y2="900" stroke="{CALR}" stroke-width="12" marker-end="url(#ac)"/>',
        f'<text x="2030" y="965" text-anchor="middle" class="t sm">tail-score ranked</text>',
        f'<text x="2735" y="865" text-anchor="middle" class="t h" fill="{CALR}">CALR</text>',
        f'<text x="2735" y="930" text-anchor="middle" class="t h" fill="{CALR}">Outputs</text>',
        '</svg>'
    ]
    path.write_text("\n".join(s), encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "plr_calr_simple.png"
    svg = OUT / "plr_calr_simple.svg"
    pdf = OUT / "plr_calr_simple.pdf"
    build_png(png)
    build_svg(svg)
    ET.parse(svg)
    Image.open(png).convert("RGB").save(pdf, "PDF", resolution=600.0)
    (OUT / "README.md").write_text(
        "# Compact PLR/CALR panel\n\n"
        "The diagram follows the current implementation while omitting capacity "
        "and threshold details. PLR combines current native-map checking, "
        "raw-mean-score ranking, internal AABB NMS, and the Top-12 output budget "
        "under `Native Check + Ranked Dedup.` CALR keeps the score mapping as "
        "the small `tail-score ranked` output-arrow annotation rather than a "
        "separate module. Confirmed PLR medoids and CALR "
        "exemplars remain revisable when later evidence arrives; explain this in "
        "the caption or method text rather than adding another loop to the figure.\n",
        encoding="utf-8",
    )
    print(png)
    print(svg)
    print(pdf)


if __name__ == "__main__":
    main()
