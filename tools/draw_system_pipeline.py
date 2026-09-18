"""Publication pipeline figure: the improved system over the BoxFusion backbone.

Backbone stages (original BoxFusion) are gray; our additions are colored by
evidence polarity: M1 dual-depth candidate recovery (green), M2 consensus
re-scoring (blue), M5 negative-evidence retirement (red), configuration-layer
releases (dashed purple). All annotated gains are the measured sequential
increments of the unified stack (dual-source M1 -> M2nl -> tier2; M5 evaluated
in the dynamic chapter). English-only labels for the paper. Pure matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

GRAY_F, GRAY_E = '#f2f2f2', '#616161'
GREEN_F, GREEN_E = '#e8f5e9', '#2e7d32'
ORANGE_F, ORANGE_E = '#fff3e0', '#e65100'
BLUE_F, BLUE_E = '#e3f2fd', '#1565c0'
RED_F, RED_E = '#ffebee', '#c62828'


def box(ax, x, y, w, h, text, fc, ec, fs=8.5, ls='-', weight='normal'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                fc=fc, ec=ec, lw=1.4, linestyle=ls, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, zorder=4, fontweight=weight, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color='#424242', style='-|>', lw=1.3,
          connectionstyle='arc3,rad=0', ls='-'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, lw=lw, color=color,
                                 connectionstyle=connectionstyle,
                                 linestyle=ls, zorder=2))


def note(ax, x, y, text, color='#37474f', fs=7.2, ha='center', weight='normal'):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fs, color=color,
            zorder=4, linespacing=1.3, fontweight=weight)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(15.2, 9.6))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 9.6)
    ax.axis('off')

    ax.text(7.6, 9.32, 'Training-Free Online Open-Vocabulary 3D Detection: '
             'Evidence-Lifecycle Pipeline (ours on top of BoxFusion)',
            ha='center', fontsize=12.5, fontweight='bold')

    # ---------------- backbone (original pipeline, gray) ----------------
    y0, h = 4.35, 0.85
    box(ax, 0.25, y0, 1.30, h, 'RGB-D\nkeyframes', GRAY_F, GRAY_E)
    box(ax, 1.85, y0, 1.55, h, 'CuTR 2D detector\n+ CLIP (frozen)', GRAY_F, GRAY_E)
    box(ax, 3.70, y0, 1.60, h, 'Top-K + Boxer\n3D lifting (frozen)', GRAY_F, GRAY_E)
    box(ax, 5.60, y0, 1.05, h, '3D NMS', GRAY_F, GRAY_E)
    box(ax, 6.95, y0, 1.35, h, '2D matching /\nassociation', GRAY_F, GRAY_E)
    box(ax, 8.60, y0, 1.65, h, 'Multi-view box\nfusion (PFO)', GRAY_F, GRAY_E)
    box(ax, 10.55, y0, 1.45, h, 'Incremental\n3D instance map', GRAY_F, GRAY_E)
    box(ax, 12.30, y0, 1.40, h, 'M2 consensus\nre-scoring', BLUE_F, BLUE_E, weight='bold')
    box(ax, 14.00, y0, 1.00, h, 'Final\noutput', GRAY_F, GRAY_E)
    for x1, x2 in ((1.55, 1.85), (3.40, 3.70), (5.30, 5.60), (6.65, 6.95),
                   (8.30, 8.60), (10.25, 10.55), (12.00, 12.30), (13.70, 14.00)):
        arrow(ax, x1, y0 + h / 2, x2, y0 + h / 2)

    # threshold release (configuration layer) on the backbone
    note(ax, 4.50, 3.82, 'score threshold release\n0.4 → 0.15  (CA-1M +7.95 AP15)',
         color='#6a1b9a', fs=7.4, weight='bold')
    arrow(ax, 4.50, 4.05, 4.50, 4.35, color='#6a1b9a', ls='--')

    # ---------------- upper: WeDetect row (tier 1) + tier-2 row ----------------
    y1 = 6.55
    box(ax, 1.85, y1, 1.55, 0.85, 'WeDetect\n(frozen, class-agnostic)', GREEN_F, GREEN_E)
    arrow(ax, 0.90, y0 + h, 0.90, y1 + 0.42, color='#2e7d32')
    arrow(ax, 0.90, y1 + 0.42, 1.85, y1 + 0.42, color='#2e7d32')
    box(ax, 3.70, y1, 1.60, 0.85, 'residual 2D instances\n(CuTR blind spots)', GREEN_F, GREEN_E, fs=7.6)
    box(ax, 5.60, y1, 2.20, 0.85, 'M1 causal funnel\n≥3 frames → birth\n(M1d size pricing)', GREEN_F, GREEN_E, fs=7.6, weight='bold')
    arrow(ax, 3.40, y1 + 0.42, 3.70, y1 + 0.42, color='#2e7d32')
    arrow(ax, 5.30, y1 + 0.42, 5.60, y1 + 0.42, color='#2e7d32')
    arrow(ax, 6.70, y1, 9.10, y0 + h, color='#2e7d32',
          connectionstyle='arc3,rad=-0.12')

    y2row = 8.15
    box(ax, 3.70, y2row, 1.60, 0.72, 'pre-NMS dense anchors\ntop-M by raw score', ORANGE_F, ORANGE_E, fs=7.6)
    arrow(ax, 3.40, y1 + 0.70, 3.70, y2row + 0.40, color='#e65100',
          connectionstyle='arc3,rad=0.25')
    box(ax, 5.60, y2row, 1.55, 0.72, 'Boxer lift +\n0.3 m voxel cluster', ORANGE_F, ORANGE_E, fs=7.6)
    arrow(ax, 5.30, y2row + 0.36, 5.60, y2row + 0.36, color='#e65100')
    box(ax, 7.45, y2row, 1.85, 0.72, 'score-ranked births\n@ flat 0.05 (no judging)', ORANGE_F, ORANGE_E, fs=7.6)
    arrow(ax, 7.15, y2row + 0.36, 7.45, y2row + 0.36, color='#e65100')
    arrow(ax, 8.35, y2row, 9.42, y0 + h, color='#e65100',
          connectionstyle='arc3,rad=-0.15')

    # ---------------- lower branch: NMS-child recycle + M5 ----------------
    y2 = 2.55
    box(ax, 5.35, y2, 2.10, 0.85, 'swallowed NMS children\n→ recovery pool', GREEN_F, GREEN_E, fs=7.6)
    arrow(ax, 6.12, y0, 6.40, y2 + 0.85, color='#2e7d32',
          connectionstyle='arc3,rad=0.2')
    box(ax, 7.75, y2, 2.05, 0.85, 'funnel ≥2 frames\n→ recovered birth', GREEN_F, GREEN_E, fs=7.6)
    arrow(ax, 7.45, y2 + 0.42, 7.75, y2 + 0.42, color='#2e7d32')
    arrow(ax, 8.78, y2 + 0.85, 9.30, y0, color='#2e7d32',
          connectionstyle='arc3,rad=0.12')

    box(ax, 10.40, y2, 2.30, 0.95, 'M5 negative-evidence\nretirement (×0.3 demote)', RED_F, RED_E, fs=7.6, weight='bold')
    arrow(ax, 11.28, y0, 11.28, y2 + 0.95, color='#c62828',
          connectionstyle='arc3,rad=0', style='<|-|>')
    note(ax, 13.35, y2 + 0.48, 'dynamic:\nstale 96% → 56%\nretired ×4.5\ndyn-AP +1.8',
         color='#c62828', fs=7.2)

    # ---------------- measured gains ----------------
    note(ax, 4.55, 5.72, 'M1 tier 1 (dual source):\nScanNet +4.00  |  CA-1M +1.16  (AP15)',
         color='#2e7d32', fs=7.6, weight='bold')
    note(ax, 8.35, 7.78, 'M1 tier 2 (anchor recycling):  ScanNet +2.05  |  CA-1M +1.43  (AP15, unified M300)',
         color='#e65100', fs=7.6, weight='bold')
    note(ax, 13.00, 5.62, 'M2 nativelogit:\nScanNet +2.22  |  CA-1M +0.12/+0.95 (AP15/AP50)',
         color='#1565c0', fs=7.4, weight='bold')

    # ---------------- legend (two rows) + disclosure ----------------
    def swatch(x, y, fc, ec, label):
        ax.add_patch(FancyBboxPatch((x, y), 0.42, 0.20, boxstyle='round,pad=0.02',
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + 0.56, y + 0.10, label, va='center', fontsize=7.6)

    swatch(0.35, 1.30, GRAY_F, GRAY_E, 'original BoxFusion backbone (frozen models, online)')
    swatch(5.60, 1.30, GREEN_F, GREEN_E, 'M1: positive-evidence birth (dual-depth recovery)')
    swatch(10.70, 1.30, ORANGE_F, ORANGE_E, 'M1 tier 2: self-discarded anchor recycling')
    swatch(0.35, 0.82, BLUE_F, BLUE_E, 'M2: cross-detector consensus ranking')
    swatch(5.60, 0.82, RED_F, RED_E, 'M5: negative-evidence retirement')
    note(ax, 0.35, 1.86, 'gains = measured sequential increments of the unified stack '
         '(dual-source M1 → M2 nativelogit → tier 2, unified M300); development-set '
         'disclosure applies; 15 FPS constraint preserved (tier 2 adds no new detector forward)',
         fs=7.0, ha='left')

    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / f'fig_pipeline.{suffix}', dpi=220,
                    bbox_inches='tight')
    print(f'wrote {args.output}/fig_pipeline.png and .pdf')


if __name__ == '__main__':
    main()
