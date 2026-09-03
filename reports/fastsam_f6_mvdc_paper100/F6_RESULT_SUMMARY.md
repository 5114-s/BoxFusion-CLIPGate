# F6 paper100 result summary

Date: 2026-08-29 (Asia/Shanghai)

## Outcome

F6 passed its complete no-GT integrity, causality, determinism, bounded-state,
and replay-runtime gates, but failed the frozen one-shot geometry-capacity
decision.  The required decision is therefore:

`discard_f6_multiview_selector_for_plus10_route`

F6 remains shadow-only.  It produced no detection, changed no native box, and
did not change the actual constant-score paper100 AP:

| Metric | Actual native AP |
|---|---:|
| AP15 | 31.0130259031 |
| AP25 | 26.7911284298 |
| AP50 | 12.0668518301 |

## Frozen one-shot capacity result

The values below use GT to select a different constructive suffix at each IoU
threshold.  They are nondeployable oracle values, not one shared prediction
list and not actual F6 AP.

| IoU | Native max matches | Native + selected max matches | Extra matches | Required | Oracle suffix AP | Oracle delta AP | Required |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.15 | 873 | 1,250 | +377 | +144 | 50.1222125629 | +19.1091866598 | +10 |
| 0.25 | 810 | 1,070 | +260 | +144 | 39.8813869733 | +13.0902585436 | +10 |
| 0.50 | 548 | 614 | **+66** | **+144** | 14.7928344831 | **+2.7259826530** | **+10** |

The 0.15 and 0.25 gates pass, but both frozen 0.50 gates fail.  The protocol
requires both gates at all three thresholds, so no active birth or deployment
is authorized and F6 must not be retuned and reevaluated on paper100.

## What can actually be attributed to F6

F6 retained Hbase for 52,134 of 52,299 sources and switched only 165 sources
across 65 scenes.  Comparing the exact F6 selection with an all-Hbase
counterfactual gives:

| IoU | All-Hbase union matches | F6-selected union matches | Net effect of 165 switches |
|---:|---:|---:|---:|
| 0.15 | 1,250 | 1,250 | 0 |
| 0.25 | 1,064 | 1,070 | +6 |
| 0.50 | 611 | 614 | +3 |

Thus almost all apparent pool capacity comes from the already-existing base
geometries.  The new past-only multi-view selector itself contributes only
0/6/3 maximum matches at IoU 0.15/0.25/0.50.

## Why another hand-written selector is not the next step

F4's GT-selected four-hypothesis upper bound has AP50 delta only
`+10.5851495498`, with 217 of its 260 chosen AP50 suffix geometries coming
from Boxer HB.  F5 retained 91 AP50 extra matches and `+3.6386990391` oracle
AP; F6 retained only 66 and `+2.7259826530`.  A perfect selector over the
current pool has too little margin above the +10 target, while both GT-free
selectors discard most useful high-IoU HB alternatives.

The next preregistered experiment should therefore expand or improve the
high-IoU candidate geometry pool, not activate F6 birth and not add another
confirmation gate.  A suitable new branch is a frozen, generally pretrained
past-only video-mask model (for example SAM2 automatic-mask/keyframe seeding
plus causal mask propagation) followed by RGB-D multi-view voxel/TSDF lifting
and the existing frozen Boxer/robust-OBB hypotheses.  It should first run as a
no-output shadow and must demonstrate an AP50 oracle ceiling comfortably above
+10 before any active birth experiment.

## Reproducibility seals

- F6 no-GT receipt SHA-256:
  `1a9f701214fe2ee9de3ea3b3a106064dc5670de5110c5c5056d622949f863727`
- F6 evaluation protocol SHA-256:
  `390b7b704b200b22ccc7e604d6b6992b0a9a99ba7c35e106c2465928b1d03e2a`
- Evaluator source SHA-256:
  `a4345243607ee75b0bfb8f635cc2d1a5b2b8d66cbedf0f979ba33b7f6953d634`
- Formal result SHA-256:
  `15c32cafd74f89bd4f67d9e51f4b2f64568025f850b2d4eef7df4450c1256010`
- Formal evaluation inputs before/after: byte-identical.
- Regression tests: 46 passed.
