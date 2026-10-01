"""Overlay tier-2 additions onto the user's original pipeline_figure.png.

The original image is loaded as the background and only NEW elements are
drawn on top (pixel coordinates, origin top-left, y down, from a vision
analysis of the original layout):
  - panel (a) lower empty area (y~370-625): the M1-tier2 anchor-recycling
    boxes (orange family, star-marked like the user's red additions);
  - a bypass arrow from tier-2 births to the global-map result box that
    crosses panel (b)'s empty bottom strip (no funnel, no size pricing);
  - the result box text refreshed to the unified-stack final numbers.
Everything else in the original image is preserved pixel-for-pixel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

ORANGE_E = '#e65100'
ORANGE_F = '#fff3e0'
PURPLE = '#6a1b9a'


def box(ax, x, y, w, h, text, fc, ec, fs=6.0, weight='normal', tc='#212121',
        lw=1.3):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=2',
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, zorder=4, fontweight=weight, linespacing=1.5, color=tc)


def arrow(ax, x1, y1, x2, y2, color, lw=1.4, style='-|>', rad=0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=9, lw=lw, color=color,
                                 connectionstyle=f'arc3,rad={rad}', zorder=2))


def note(ax, x, y, text, color='#37474f', fs=5.2, ha='center', weight='normal'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=color,
            zorder=5, linespacing=1.4, fontweight=weight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path,
                        default=Path('/data/ZhaoX/BoxFusion/docs/pipeline_figure.png'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = plt.imread(args.base)
    height, width = image.shape[:2]
    fig = plt.figure(figsize=(width / 220, height / 220), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, extent=[0, width, height, 0], aspect='auto', zorder=1)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')

    # ---- panel (a) lower empty area: tier-2 branch ----
    box(ax, 245, 392, 360, 100,
        '★ M1-tier2 pre-NMS 锚级回收（新·配置层）\n'
        'WeDetect dense 头未截断锚（1e-5 尾部）→ 原始分数 top-M\n'
        '→ 复用 M1a Boxer 提升（第四流）→ 0.3 m 体素跨帧去重',
        ORANGE_F, ORANGE_E, fs=5.8, weight='bold')
    box(ax, 245, 522, 360, 82,
        '分数排序出生 @ flat 0.05（不判断、只补位）\n'
        '尾部扩展＋几何拯救：ScanNet +2.05 ｜ CA-1M +1.43（AP15）',
        ORANGE_F, ORANGE_E, fs=5.8, weight='bold')
    arrow(ax, 425, 392, 425, 372, color=ORANGE_E)
    arrow(ax, 425, 492, 425, 522, color=ORANGE_E)

    # ---- bypass route: tier-2 births -> global map result box ----
    arrow(ax, 605, 563, 1058, 563, color=ORANGE_E, style='-')
    arrow(ax, 1058, 563, 1088, 588, color=ORANGE_E)
    note(ax, 855, 548, 'tier-2 出生追加（绕过 b：无漏斗、无尺寸定价）',
         color=ORANGE_E, fs=5.2, weight='bold')

    # ---- refresh the result box (old pre-tier2 numbers) ----
    ax.add_patch(FancyBboxPatch((1092, 557), 316, 66, boxstyle='round,pad=2',
                                fc='#1a1a1a', ec='#ffffff', lw=1.0, zorder=6))
    ax.text(1250, 578, '全局 3D 对象框（统一栈 + tier-2）', ha='center',
            va='center', fontsize=5.4, color='#ffffff', zorder=7,
            fontweight='bold')
    ax.text(1250, 598, 'ScanNet 43.35 / 39.15 / 19.56（M300 统一）', ha='center',
            va='center', fontsize=5.4, color='#ff8a65', zorder=7)
    ax.text(1250, 614, 'CA-1M 47.56 / 39.58 / 17.63', ha='center',
            va='center', fontsize=5.4, color='#ff8a65', zorder=7)

    fig.savefig(args.output, dpi=220)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
