"""Truncation-signal computation for the dynamic policy front-end.

Causal, GT-free: a mask is truncated from below (resp. above) when pixels
just beyond its lower (resp. upper) boundary -- inside the detection box
columns -- lie NEARER than the person's own median depth by a margin
(a real occluder in front), or when the mask touches the image border on
that side.  Side (left/right) crops are reported but not completed by the
current height-only prior.
"""
from __future__ import annotations
import numpy as np

OCCLUDER_MARGIN_M = 0.15
OCCLUDER_FRACTION = 0.5
BAND_PX = 12


def truncation_signals(mask, depth, box2d, person_depth, img_wh=(640, 480)):
    """mask: bool HxW; depth: metres HxW; box2d: [x1,y1,x2,y2];
    person_depth: median depth of the masked points."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return {'trunc_below': False, 'trunc_above': False,
                'border': {'l': False, 'r': False, 't': False, 'b': False}}
    x1, y1, x2, y2 = box2d
    col_lo = int(max(xs.min(), x1))
    col_hi = int(min(xs.max(), x2))
    top, bot = int(ys.min()), int(ys.max())

    def occluder_fraction(rows):
        rows = [r for r in rows if 0 <= r < H]
        if not rows:
            return 0.0
        band = depth[rows, col_lo:col_hi + 1]
        valid = band[(band > 0.05) & (band < 12.0)]
        if len(valid) < 20:
            return 0.0
        return float(np.mean(valid < person_depth - OCCLUDER_MARGIN_M))

    f_below = occluder_fraction(range(bot + 2, bot + 2 + BAND_PX))
    f_above = occluder_fraction(range(top - 1 - BAND_PX, top - 1))
    border_b = bot >= H - 2 or y2 >= H - 1
    border_t = top <= 1 or y1 <= 0
    # Completion fires ONLY on the occluder signature (a NEARER surface just
    # beyond the mask boundary): this is the scenario class the 0.6 visibility
    # prior was validated on.  Border crops are reported but NOT completed --
    # their true visible fraction is unknown and 0.6 would over-extend.
    return {
        'trunc_below': bool(f_below >= OCCLUDER_FRACTION),
        'trunc_above': bool(f_above >= OCCLUDER_FRACTION),
        'border': {'l': bool(xs.min() <= 1), 'r': bool(xs.max() >= W - 2),
                   't': bool(border_t), 'b': bool(border_b)},
        'occluder_fraction_below': round(f_below, 3),
        'occluder_fraction_above': round(f_above, 3),
    }
