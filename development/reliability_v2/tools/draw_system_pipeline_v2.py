"""System figure v2: user's (a)(b)(c) panel layout, updated to the final ledger.

Base structure (user's improved BoxFusion figure): gray = untouched original
components; red-star boxes = our additions; shared components marked. Versus
the user's printout this adds M1 tier 2 (pre-NMS anchor recycling) in panel
(a) with its score-ranked low-price birth path, plus the score-threshold
release note, and updates all gain annotations to the unified stack
(dual-source M1 -> M2 nativelogit -> tier 2). M5 is intentionally excluded
(reported separately in the dynamic chapter). Chinese labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
matplotlib.rcParams['axes.unicode_minus'] = False

GRAY_F, GRAY_E = '#f5f5f5', '#757575'
PANEL = {'a': '#eef4fb', 'b': '#fdf8e7', 'c': '#fdeeee'}
RED_F, RED_E = '#fdecea', '#c62828'
GREEN_F, GREEN_E = '#e8f5e9', '#2e7d32'
ORANGE_F, ORANGE_E = '#fff3e0', '#e65100'
BLUE_F, BLUE_E = '#e3f2fd', '#1565c0'
PURPLE = '#6a1b9a'


def box(ax, x, y, w, h, text, fc='#ffffff', ec='#616161', fs=8.0, star=False,
        weight='normal', dashed=False, tc='#212121'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                fc=fc, ec=ec, lw=1.5 if star else 1.1,
                                linestyle='--' if dashed else '-', zorder=3))
    prefix = '★ ' if star else ''
    ax.text(x + w / 2, y + h / 2, prefix + text, ha='center', va='center',
            fontsize=fs, zorder=4, fontweight=weight, linespacing=1.35, color=tc)


def arrow(ax, x1, y1, x2, y2, color='#424242', lw=1.2, style='-|>',
          connectionstyle='arc3,rad=0', dashed=False):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, lw=lw, color=color,
                                 connectionstyle=connectionstyle,
                                 linestyle='--' if dashed else '-', zorder=2))


def note(ax, x, y, text, color='#37474f', fs=6.8, ha='center', weight='normal'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=color,
            zorder=5, linespacing=1.3, fontweight=weight)


PANEL_EDGES = {'a': '#5b8db8', 'b': '#c9a227', 'c': '#c96a6a'}


def panel(ax, x, y, w, h, tag, title):
    edge = PANEL_EDGES[tag]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                fc=PANEL[tag], ec=edge, lw=1.6, zorder=1, alpha=0.55))
    ax.text(x + 0.12, y + h - 0.16, f'({tag}) {title}', fontsize=10,
            fontweight='bold', color=edge, zorder=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.4, 8.55))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    panel(ax, 0.25, 1.55, 4.85, 7.30, 'a', 'Proposal Generation')
    panel(ax, 5.30, 1.55, 4.35, 7.30, 'b', 'Association Module')
    panel(ax, 9.85, 1.55, 6.30, 7.30, 'c', 'Multi-View Box Fusion')

    # -------- panel (a): generation --------
    box(ax, 0.55, 7.05, 1.95, 0.95, 'CuTR（原生）\n单视角 3D proposal\n+ CLIP 语义',
        GRAY_F, GRAY_E, dashed=True)
    box(ax, 2.90, 7.05, 2.00, 0.95, 'WeDetect-Uni\n第二前向（新）\n类无关 proposal', RED_F, RED_E, star=True)
    box(ax, 0.55, 5.55, 4.35, 1.00, 'M1a BoxerNet 几何引擎（共享）\n同一流水线四流复用：\nnative ＋ child ＋ WeDetect ＋ 锚级',
        GREEN_F, GREEN_E, star=True, fs=7.6)
    note(ax, 2.72, 5.18, 'feature cache：每帧编码一次，多路查询批量解码', fs=6.6)
    box(ax, 0.55, 3.55, 4.35, 1.00, '★ M1-tier2 pre-NMS 锚级回收（新·配置层）\ndense 锚按原始分数取 top-M（1e-5 尾部）\n→ 共享 Boxer 提升 → 0.3 m 体素跨帧去重',
        ORANGE_F, ORANGE_E, star=True, fs=7.4)
    box(ax, 0.55, 2.05, 4.35, 0.85, '分数排序出生 @ flat 0.05（不判断、只补位）\n尾部扩展 + 几何拯救：ScanNet +2.05 ｜ CA-1M +1.43',
        ORANGE_F, ORANGE_E, star=True, fs=7.4, weight='bold')
    arrow(ax, 1.52, 7.05, 1.52, 6.55)
    arrow(ax, 3.90, 7.05, 3.90, 6.55, color='#2e7d32')
    arrow(ax, 2.72, 5.55, 2.72, 4.55, color='#e65100')
    arrow(ax, 2.72, 3.55, 2.72, 2.90, color='#e65100')
    # tier-2 births bypass panel (b) entirely: no funnel, no size pricing.
    arrow(ax, 2.72, 2.05, 2.72, 1.82, color='#e65100', style='-')
    arrow(ax, 2.72, 1.82, 10.30, 1.82, color='#e65100', style='-')
    arrow(ax, 10.30, 1.82, 10.30, 2.10, color='#e65100')
    note(ax, 6.60, 1.98, 'tier-2 追加（绕过 b：无漏斗、无尺寸定价，flat 0.05）',
         color='#e65100', fs=6.6)
    note(ax, 1.52, 8.32, '阈值释放 0.4 → 0.15\n（CA-1M +7.95 AP15，配置层）',
         color=PURPLE, fs=6.8, weight='bold')

    # -------- panel (b): association --------
    box(ax, 5.55, 7.05, 3.85, 0.95, '3D NMS + 2D Matching（原生）\nSA / CA 关联逻辑未动', GRAY_F, GRAY_E, dashed=True)
    box(ax, 5.55, 5.55, 3.85, 0.95, '★ M1b child 回收缓冲（新）\n被 NMS 吞掉的 child 框保存 → 第二路候选', RED_F, RED_E, star=True, fs=7.6)
    box(ax, 5.55, 3.90, 3.85, 1.20, '★ M1c 因果确认漏斗（新）\n出生 = ≥3（WeDetect）/ ≥2（child）个不同历史帧\n独立支持；4.7 万候选 → ~1.1k birth（42× 压缩）', RED_F, RED_E, star=True, fs=7.4)
    box(ax, 5.55, 2.10, 3.85, 1.05, '★ M1d 尺寸定价（新）\nbirth 按最大边长查表定价\n（标注策略可从几何读出）', RED_F, RED_E, star=True, fs=7.4)
    arrow(ax, 7.48, 7.05, 7.48, 6.50, color='#c62828')
    arrow(ax, 7.48, 5.55, 7.48, 5.10, color='#c62828')
    arrow(ax, 7.48, 3.90, 7.48, 3.15, color='#c62828')
    note(ax, 7.48, 6.78, '吞并事件', fs=6.6, color='#c62828')

    # -------- panel (c): fusion / output --------
    box(ax, 10.10, 7.05, 5.80, 0.95, 'Multi-View Box Fusion / PFO（原生）\n跨视角投影一致性优化中心与尺寸 —— 本路线未改动', GRAY_F, GRAY_E, dashed=True)
    box(ax, 10.10, 4.90, 5.80, 1.30, '★ M2 跨检测器共识重排（新）\n最近邻投影 × WeDetect 所见 → 支持度\nScanNet 加性 +2.22 ｜ CA-1M 保序 logit +0.12/+0.95\nbirth 分数冻结；CA-1M 上对 tier-2 追加行按位置不变性免交互',
        BLUE_F, BLUE_E, star=True, fs=7.2)
    box(ax, 10.10, 2.10, 5.80, 1.35, '全局 3D 对象框（统一栈终局）\nScanNet 43.35 / 39.15 / 19.56（M300 统一配置；M150 最优 43.71/39.37/19.76）\nCA-1M 47.56 / 39.58 / 17.63   vs 论文 +14.1/+14.5/+11.5 ｜ +16.4/+14.1/+8.8',
        '#e8eaf6', '#3949ab', fs=7.8, weight='bold')
    arrow(ax, 2.50, 7.52, 5.55, 7.52, color='#9e9e9e', dashed=True)
    note(ax, 4.00, 7.72, 'native proposal 流', fs=6.6)
    arrow(ax, 4.90, 7.52, 5.55, 6.10, color='#2e7d32',
          connectionstyle='arc3,rad=-0.25')
    note(ax, 4.42, 6.68, 'WeDetect 发现流', fs=6.6, color='#2e7d32')
    arrow(ax, 11.00, 7.05, 11.00, 6.20)
    arrow(ax, 13.00, 4.90, 13.00, 3.45)

    # cross-panel flows
    arrow(ax, 9.40, 2.62, 10.10, 2.62, color='#c62828')
    note(ax, 9.75, 2.88, 'birth 入图', fs=6.4, color='#c62828')

    # top reuse annotation
    note(ax, 8.20, 8.95, 'WeDetect 提案：M1a 提升（建模）+ M2 加分（排序）——一次前向三处复用；'
         'tier 2 复用 dense 头与 Boxer，全程零新增前向', color='#c62828', fs=8.0, weight='bold')

    # legend
    lx, ly = 0.45, 1.28
    box(ax, lx, ly, 0.40, 0.22, '', GRAY_F, GRAY_E)
    note(ax, lx + 0.52, ly + 0.11, '原版组件（未改动）', fs=7.0, ha='left')
    box(ax, lx + 2.20, ly, 0.40, 0.22, '', RED_F, RED_E)
    note(ax, lx + 2.72, ly + 0.11, '★ 新增模块', fs=7.0, ha='left')
    box(ax, lx + 4.10, ly, 0.40, 0.22, '', GREEN_F, GREEN_E)
    note(ax, lx + 4.62, ly + 0.11, '共享/回收（M1 双深度）', fs=7.0, ha='left')
    box(ax, lx + 6.60, ly, 0.40, 0.22, '', ORANGE_F, ORANGE_E)
    note(ax, lx + 7.12, ly + 0.11, 'tier 2 锚级回收（配置层）', fs=7.0, ha='left')
    box(ax, lx + 9.60, ly, 0.40, 0.22, '', BLUE_F, BLUE_E)
    note(ax, lx + 10.12, ly + 0.11, 'M2 排序', fs=7.0, ha='left')
    note(ax, 0.45, 0.72, '贡献汇总（AP15，统一栈：双源 M1 → M2 nativelogit → tier 2）：'
         'M1 双源 +4.00/+1.16（ScanNet/CA-1M）｜ tier 2 +2.05/+1.43 ｜ M2 +2.22/+0.12 ｜ '
         '阈值释放 +7.95（CA-1M）。'
         '底座（real-score + Top-K + Boxer）：ScanNet 35.04 ｜ CA-1M 36.90；开发集调优已披露。',
         fs=7.2, ha='left')

    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / f'fig_pipeline_v2.{suffix}', dpi=220,
                    bbox_inches='tight')
    print(f'wrote {args.output}/fig_pipeline_v2.png and .pdf')


if __name__ == '__main__':
    main()
