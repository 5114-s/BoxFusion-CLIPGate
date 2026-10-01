#!/usr/bin/env python3
"""Final-system pipeline figure in the reference style: input panel, lettered
method panels (a)-(f), output panel, and a numbered 7-step bottom strip.

Style borrows the user's reference figure: panel families color-coded
(blue=native detection, orange=evidence recovery M1, yellow=discipline,
pink=re-ranking/readout, green=output), trapezoid+frost symbol = frozen
network, dashed arrows = auxiliary/evidence flows, English labels.
All content matches reports/paper_main_table_20260916 and the frozen recipe.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Circle, Rectangle

matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

W, H = 1760, 780
BLUE_F, BLUE_E = '#e3f2fd', '#1565c0'
ORNG_F, ORNG_E = '#fff3e0', '#e65100'
YELL_F, YELL_E = '#fffde7', '#f9a825'
PINK_F, PINK_E = '#fce4ec', '#c2185b'
GREEN_F, GREEN_E = '#e8f5e9', '#2e7d32'
GRAY = '#546e7a'
DARK = '#212121'


def rbox(ax, x, y, w, h, text='', fc='#ffffff', ec='#616161', fs=7.0,
         weight='normal', lw=1.2, tc=DARK, ls='-', z=3, radius=3, align='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f'round,pad={radius},rounding_size=3',
                                fc=fc, ec=ec, lw=lw, zorder=z, linestyle=ls))
    if text:
        ax.text(x + w / 2 if align == 'center' else x + 6,
                y + h / 2, text, ha=align, va='center', fontsize=fs,
                zorder=z + 1, fontweight=weight, linespacing=1.45, color=tc)


def trap(ax, x, y, w, h, text, fc, ec, fs=7.0, z=3):
    """Trapezoid = frozen network module (reference convention)."""
    k = h * 0.28
    pts = [(x, y), (x + w, y + k), (x + w, y + h - k), (x, y + h)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec=ec, lw=1.3, zorder=z))
    ax.text(x + w / 2 + k / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, zorder=z + 1, fontweight='bold', linespacing=1.4,
            color=DARK)


def arrow(ax, x1, y1, x2, y2, color='#37474f', lw=1.6, style='-|>', rad=0.0,
          ls='-', z=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=lw, color=color,
                                 linestyle=ls,
                                 connectionstyle=f'arc3,rad={rad}', zorder=z))


def note(ax, x, y, text, color='#37474f', fs=6.4, ha='center', weight='normal'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=color,
            zorder=5, linespacing=1.4, fontweight=weight)


def panel(ax, x, y, w, h, tag, title, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=4,rounding_size=6',
                                fc=fc, ec=ec, lw=1.6, zorder=1))
    ax.text(x + 12, y + 16, f'({tag})' if tag else '', fontsize=10,
            fontweight='bold', color=ec, zorder=2, va='center')
    ax.text(x + (38 if tag else 14), y + 16, title, fontsize=9,
            fontweight='bold', color=ec, zorder=2, va='center')


def photo(ax, x, y, w, h, seed=0):
    """Stylized camera frame (photo placeholder)."""
    rng = np.random.default_rng(seed)
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=1.5,rounding_size=2',
                                fc='#eceff1', ec='#78909c', lw=1.0, zorder=3))
    for _ in range(5):
        cx, cy = x + w * rng.uniform(.15, .85), y + h * rng.uniform(.15, .85)
        r = h * rng.uniform(.1, .22)
        ax.add_patch(Circle((cx, cy), r, fc='#b0bec5', ec='none', zorder=4,
                            alpha=.8))


def cuboid(ax, x, y, s, fc, ec, lw=1.1, z=4, alpha=1.0):
    """Isometric 3D box glyph."""
    dx, dy = s * .42, s * .3
    front = [(x, y), (x + s, y), (x + s, y + s * .8), (x, y + s * .8)]
    top = [(x, y + s * .8), (x + dx, y + s * .8 + dy), (x + s + dx, y + s * .8 + dy), (x + s, y + s * .8)]
    side = [(x + s, y), (x + s + dx, y + dy), (x + s + dx, y + s * .8 + dy), (x + s, y + s * .8)]
    for pts, shade in ((front, fc), (top, fc), (side, fc)):
        ax.add_patch(Polygon(pts, closed=True, fc=shade, ec=ec, lw=lw,
                             zorder=z, alpha=alpha))


def snow(ax, x, y, s=7, color='#0277bd', z=6):
    """Six-armed snowflake marker (font glyph unavailable)."""
    for ang in (0, 60, 120):
        a = np.deg2rad(ang)
        dx, dy = s * np.cos(a), s * np.sin(a)
        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=1.1,
                zorder=z, solid_capstyle='round')


def camera_icon(ax, x, y, s=14, color='#455a64'):
    ax.add_patch(Rectangle((x, y), s * 1.6, s, fc='white', ec=color, lw=1.2,
                           zorder=4))
    ax.add_patch(Circle((x + s * .8, y + s * .5), s * .32, fc='none',
                        ec=color, lw=1.2, zorder=5))
    ax.add_patch(Rectangle((x + s * 1.15, y + s * 1.05), s * .35, s * .28,
                           fc=color, ec='none', zorder=5))


def kf_stack(ax, x, y, w, h, n=3, color='#8e24aa'):
    for i in range(n):
        ax.add_patch(FancyBboxPatch((x + i * 7, y + i * 7), w, h,
                                    boxstyle='round,pad=1,rounding_size=1.5',
                                    fc='white', ec=color, lw=1.1, zorder=3 + i))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')

    # ---------- title / legend ----------
    ax.text(W / 2, 30, 'Hierarchical Evidence Recovery for Training-Free '
                       'Online Open-Vocabulary 3D Detection',
            ha='center', va='center', fontsize=15, fontweight='bold')
    note(ax, W - 14, 58, '* frozen network (no training)     trapezoid = frozen NN     '
                         '— main flow     -- auxiliary / evidence flow',
         fs=6.6, ha='right', color='#607d8b')

    MY = 92       # main row top
    MH = 470      # main row height
    # ---------- input ----------
    panel(ax, 34, MY, 158, MH, '', 'Input', '#ffffff', '#263238')
    photo(ax, 52, MY + 44, 96, 60, seed=1)
    photo(ax, 64, MY + 118, 96, 60, seed=2)
    photo(ax, 76, MY + 192, 96, 60, seed=3)
    note(ax, 113, MY + 262, 'streaming RGB-D\n(every 25th frame = keyframe)',
         fs=6.4)
    camera_icon(ax, 76, MY + 316, s=12)
    camera_icon(ax, 106, MY + 316, s=12)
    note(ax, 113, MY + 356, 'camera poses\n(SLAM)', fs=6.2)
    note(ax, 113, MY + 398, 'ScanNet · CA-1M\n+ 50-scene frozen holdout',
         fs=6.2, color='#78909c')
    arrow(ax, 192, MY + MH / 2, 224, MY + MH / 2, lw=2.0)

    # ---------- (a) native detection ----------
    panel(ax, 226, MY, 286, MH, 'a', 'Native Detection · Online Fusion',
          BLUE_F, BLUE_E)
    trap(ax, 250, MY + 44, 108, 46, 'CuTR\nmono 3D', '#ffffff', BLUE_E)
    snow(ax, 298, MY + 44 - 9)
    trap(ax, 250, MY + 120, 108, 46, 'WeDetect-Uni\nopen-vocab 2D',
         '#ffffff', BLUE_E)
    snow(ax, 298, MY + 120 - 9)
    trap(ax, 250, MY + 196, 108, 46, 'Boxer\n2D→3D lift', '#ffffff',
         BLUE_E)
    snow(ax, 298, MY + 196 - 9)
    arrow(ax, 304, MY + 166, 304, MY + 190, color=BLUE_E)
    rbox(ax, 384, MY + 120, 108, 62, 'online keyframe\nfusion\n(gap 25)',
         '#ffffff', BLUE_E, fs=6.6)
    arrow(ax, 358, MY + 67, 384, MY + 140, color=BLUE_E, rad=-0.12)
    arrow(ax, 358, MY + 143, 384, MY + 152, color=BLUE_E)
    arrow(ax, 358, MY + 219, 384, MY + 165, color=BLUE_E, rad=0.12)
    rbox(ax, 250, MY + 280, 242, 60, '', '#263238', '#263238')
    ax.text(371, MY + 322, 'confident map rows  ·  score ≥ 0.5', ha='center',
            va='center', fontsize=7.2, color='white', fontweight='bold',
            zorder=5)
    cuboid(ax, 300, MY + 288, 15, '#90caf9', BLUE_E)
    cuboid(ax, 330, MY + 292, 12, '#90caf9', BLUE_E)
    cuboid(ax, 355, MY + 288, 10, '#90caf9', BLUE_E)
    note(ax, 371, MY + 368, 'per-frame proposals are NMS-truncated\nand '
                            'score-thresholded on admission', fs=6.2,
         color='#5d4037')

    # ---------- (b) M1 recovery ----------
    panel(ax, 532, MY, 372, MH, 'b', 'M1 · Hierarchical Evidence Recovery',
          ORNG_F, ORNG_E)
    # visual equation: dense anchors - survivors = tail
    rbox(ax, 550, MY + 40, 88, 40, 'dense anchors\n(pre-NMS tail)', '#ffffff',
         ORNG_E, fs=6.2)
    ax.text(648, MY + 60, '⊖', fontsize=13, ha='center', va='center', zorder=5)
    rbox(ax, 660, MY + 40, 88, 40, 'post-NMS\nsurvivors', '#ffffff', ORNG_E,
         fs=6.2)
    ax.text(758, MY + 60, '=', fontsize=11, ha='center', va='center', zorder=5)
    rbox(ax, 770, MY + 40, 114, 40, 'truncated evidence\n(recurring objects)',
         ORNG_F, ORNG_E, fs=6.2, weight='bold')
    # M1-P lane
    rbox(ax, 550, MY + 120, 334, 62, 'M1-P  proposal-level\nNMS-suppressed '
         'proposals → confirm funnel\n(≥3 frames · cap 12 · dedup 0.25)',
         '#ffffff', ORNG_E, fs=6.4)
    # M1-A lane
    rbox(ax, 550, MY + 212, 334, 84, 'M1-A  anchor-level\nraw-score top-300/frame '
         '→ Boxer lift (single pool)\n→ 0.3 m voxel state → ≥3 distinct frames '
         '→ birth', '#ffffff', ORNG_E, fs=6.4, lw=1.8)
    rbox(ax, 798, MY + 280, 80, 14, 'STRICTLY ONLINE', ORNG_E, ORNG_E, fs=5.4,
         weight='bold', tc='white')
    note(ax, 717, MY + 322, 'no future frames · bounded state · deterministic '
         'unique scores', fs=6.2, color=ORNG_E, weight='bold')
    arrow(ax, 717, MY + 88, 717, MY + 116, color=ORNG_E)
    arrow(ax, 717, MY + 184, 717, MY + 210, color=ORNG_E)
    # dashed evidence flow from (a) truncation into (b)
    arrow(ax, 512, MY + 310, 530, MY + 310, lw=2.0)
    arrow(ax, 512, MY + 300, 546, MY + 72, color=ORNG_E, ls=(0, (4, 3)))
    ax.text(524, MY + 186, 'truncation discards recurring evidence',
            fontsize=5.6, color=ORNG_E, ha='center', va='center',
            rotation=90, zorder=5)
    arrow(ax, 717, MY + 298, 717, MY + 320, color=ORNG_E)
    note(ax, 717, MY + 340, 'recovered births\n(discipline in panel c)', fs=6.4)
    arrow(ax, 886, MY + 160, 930, MY + 160, color=ORNG_E)      # to (c)
    arrow(ax, 886, MY + 170, 930, MY + 170, color=ORNG_E)

    # ---------- (c) verification & pricing ----------
    panel(ax, 932, MY, 258, 232, 'c', 'Birth Discipline', YELL_F, YELL_E)
    rows = ['✓ temporal recurrence ≥ 3 distinct frames',
            '✓ 0.3 m voxel cross-frame dedup',
            '✓ bounded state: 4,096 active / 640 births',
            '✓ tail score floor ∈ [0.04, 0.05)']
    for i, t in enumerate(rows):
        note(ax, 948, MY + 46 + i * 27, t, fs=6.4, ha='left')
    note(ax, 1061, MY + 186, 'recall tail extends —\nranking stays intact',
         fs=6.6, color=YELL_E, weight='bold')

    # ---------- (d) lifecycle ----------
    panel(ax, 932, MY + 252, 258, 218, 'd', 'Map Lifecycle · M5', YELL_F,
          YELL_E)
    rbox(ax, 948, MY + 292, 104, 40, 'in-view unsupported\n(ratio ≥ 0.8)',
         '#ffffff', YELL_E, fs=6.2)
    rbox(ax, 948, MY + 360, 104, 40, 'depth cavity\n(no core points)',
         '#ffffff', YELL_E, fs=6.2)
    rbox(ax, 1070, MY + 330, 96, 46, 'demote ×0.3\n(negative evidence)',
         '#ffffff', YELL_E, fs=6.2)
    arrow(ax, 1054, MY + 332, 1066, MY + 348, color=YELL_E)
    arrow(ax, 1054, MY + 380, 1066, MY + 366, color=YELL_E)
    note(ax, 1061, MY + 404, 'retires ghost rows', fs=6.2, color=YELL_E)

    # ---------- (e) M2 ----------
    panel(ax, 1206, MY, 236, 232, 'e', 'M2 · Consensus Re-ranking', PINK_F,
          PINK_E)
    kf_stack(ax, 1226, MY + 46, 64, 44)
    note(ax, 1258, MY + 116, 'keyframe views', fs=6.2)
    rbox(ax, 1320, MY + 46, 104, 46, 'support s′\n(exclusive matching)',
         '#ffffff', PINK_E, fs=6.2)
    arrow(ax, 1304, MY + 68, 1318, MY + 68, color=PINK_E)
    rbox(ax, 1222, MY + 150, 204, 44, 'nativelogit:  σ( logit s  +  2·s′ )',
         '#ffffff', PINK_E, fs=6.2, weight='bold')
    arrow(ax, 1324, MY + 94, 1324, MY + 148, color=PINK_E)
    note(ax, 1325, MY + 218, 'native rows re-scored · birth scores frozen',
         fs=6.2, color=PINK_E)

    # ---------- (f) semantic ----------
    panel(ax, 1206, MY + 252, 236, 218, 'f', 'Semantic Readout', PINK_F,
          PINK_E)
    rbox(ax, 1218, MY + 296, 88, 50, 'largest-projection\nauto crop',
         '#ffffff', PINK_E, fs=6.2)
    trap(ax, 1322, MY + 296, 96, 50, 'CLIP\nViT-H-14', '#ffffff',
         PINK_E, fs=6.6)
    snow(ax, 1366, MY + 296 - 9, color=PINK_E)
    arrow(ax, 1308, MY + 321, 1320, MY + 321, color=PINK_E)
    rbox(ax, 1222, MY + 374, 204, 34, 'open-vocabulary labels (18-class table)',
         '#ffffff', PINK_E, fs=6.4)
    arrow(ax, 1324, MY + 348, 1324, MY + 372, color=PINK_E)
    note(ax, 1324, MY + 428, 'terminal · training-free readout', fs=6.2,
         color=PINK_E)

    # chain: (c)->(d)->(e)->(f)
    arrow(ax, 1061, MY + 234, 1061, MY + 250, color=YELL_E)
    arrow(ax, 1190, MY + 360, 1204, MY + 176, color='#8d6e63', rad=0.12)
    arrow(ax, 1324, MY + 234, 1324, MY + 250, color=PINK_E)

    # ---------- output ----------
    panel(ax, 1462, MY, 264, MH, '', 'Output', GREEN_F, GREEN_E)
    rbox(ax, 1482, MY + 40, 224, 56, '', '#1b5e20', '#1b5e20')
    ax.text(1594, MY + 68, 'Online Open-Vocab\n3D Object Map', ha='center',
            va='center', fontsize=8.0, color='white', fontweight='bold',
            zorder=5, linespacing=1.4)
    for bx, by, sz in ((1520, MY + 108, 13), (1548, MY + 114, 9),
                       (1572, MY + 108, 11), (1600, MY + 116, 8)):
        cuboid(ax, bx, by, sz, '#a5d6a7', GREEN_E)
    # timeline: online births per keyframe; finalize at t_N
    ty = MY + 168
    ax.annotate('', xy=(1694, ty), xytext=(1486, ty),
                arrowprops=dict(arrowstyle='-|>', color='#37474f', lw=1.4))
    xs = np.linspace(1500, 1666, 7)
    for x in xs[:-1]:
        ax.add_patch(Circle((x, ty), 4.5, fc=ORNG_E, ec='none', zorder=4))
    ax.add_patch(Circle((xs[-1], ty), 4.5, fc=PINK_E, ec='none', zorder=4))
    ax.text(1484, ty + 20, 't₀', fontsize=6.5, ha='center')
    ax.text(1698, ty + 20, 't$_N$', fontsize=6.5, ha='center')
    note(ax, 1594, ty + 52, 'M1-A births appear immediately (causal) ·\n'
         'M1-P finalize / M2 re-rank complete at scene end', fs=6.0,
         color='#33691e')
    rbox(ax, 1482, MY + 262, 224, 66, 'ScanNet  35.0 → 44.1  AP15  (+9.0)\n'
         'CA-1M  44.9 → 47.3  AP15  (+2.5)\n16.7 FPS  ·  frozen-constant '
         'holdout ✓', '#ffffff', GREEN_E, fs=6.6, weight='bold', lw=1.6)
    note(ax, 1594, MY + 360, 'no training · frozen backbones ·\nbounded '
         'online state', fs=6.2, color=GREEN_E, weight='bold')
    arrow(ax, 1440, MY + 430, 1480, MY + 380, lw=2.0)
    

    # ---------- bottom 7-step strip ----------
    SY, SH = 592, 84
    steps = [
        ('1', 'Detect', 'frozen CuTR · WeDetect-Uni · Boxer', BLUE_F, BLUE_E),
        ('2', 'Recover', 'M1-P proposals + M1-A anchor tail', ORNG_F, ORNG_E),
        ('3', 'Verify', '≥3-frame recurrence · voxel dedup', YELL_F, YELL_E),
        ('4', 'Price', 'score-floor births, ranking intact', YELL_F, YELL_E),
        ('5', 'Retire', 'M5 negative-evidence demotion', YELL_F, YELL_E),
        ('6', 'Re-rank', 'M2 cross-view consensus', PINK_F, PINK_E),
        ('7', 'Map & Read', 'bounded online map + CLIP labels', GREEN_F,
         GREEN_E),
    ]
    n = len(steps)
    total = W - 68
    bw = (total - (n - 1) * 10) / n
    for i, (num, name, sub, fc, ec) in enumerate(steps):
        x = 34 + i * (bw + 10)
        rbox(ax, x, SY, bw, SH, '', fc, ec, lw=1.4)
        ax.add_patch(Circle((x + 18, SY + 24), 11, fc=ec, ec='none', zorder=4))
        ax.text(x + 18, SY + 24, num, fontsize=9, color='white',
                ha='center', va='center', fontweight='bold', zorder=5)
        ax.text(x + 36, SY + 24, name, fontsize=8.4, fontweight='bold',
                ha='left', va='center', color=DARK, zorder=5)
        ax.text(x + bw / 2, SY + 60, sub, fontsize=5.9, ha='center',
                va='center', color='#455a64', zorder=5)
    note(ax, W / 2, SY + SH + 26, 'Training-free: every learned component is '
         'frozen; M1/M2/M5 are deterministic post-processing.  M1-A is '
         'strictly online; M1-P finalize, M2 and the semantic readout are '
         'scene-end stages.', fs=6.6, color='#607d8b')

    fig.savefig(args.output, dpi=200, facecolor='white')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
