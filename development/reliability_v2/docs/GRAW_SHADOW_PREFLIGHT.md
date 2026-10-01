# Graw-shadow to Graw-active preflight

Date: 2026-08-23 (Asia/Shanghai)

## Frozen protocol

- Base arm: T05 (`score_thresh=0.5`, native scores preserved, appearance gate
  disabled, Reliable-View Top-K enabled).
- Ordered scenes: `scene0568_00`, `scene0606_01`, `scene0377_02`.
- CuTR proposals are replayed from the sealed E2 cache
  `scannet-graw-e2-score05-preflight3-v3-r1`.
- Graw is training-free, causal, begin-frame-past only, and bounded to 64 new
  proposals, 1,024 tracks, five views per track, 512 voxels per view and 1,024
  union voxels per track.
- Raw fragments use valid depth and signed 5 cm world voxels.  No SMOV edge,
  depth-jump or component cleaning is applied in this experiment.
- Graw-shadow observes native unmatched-retained proposals but cannot alter
  native association, BoxFusion, CLIP classification, scores or output rows.

## Shadow noninterference and online cost

All three observer traces are valid.  There are no observer errors, native/RNG
boundary violations, or matcher fail-open events.  Compared with both sealed
E2 replay controls, all 46 terminal rows keep exact scene order, row order,
class and score.  The largest shadow-vs-control terminal corner difference is
0.345 mm, within the previously measured replay stochastic floor.

| quantity | result |
|---|---:|
| keyframes | 199 |
| native unmatched-retained candidates | 218 |
| counterfactual associations | 4 (1.83%) |
| fragment commits / abstentions | 1,106 / 13 |
| matcher fail-open | 0 |
| observer latency p50 | 3.474 ms/keyframe |
| observer latency p95 | 6.687 ms/keyframe |
| observer latency maximum | 23.051 ms/keyframe |

The shadow branch therefore passes the noninterference, causality, boundedness
and online-latency checks.  As a shadow module, its AP is intentionally
identical to T05: AP15 26.8814, AP25 26.8814 and AP50 18.1330 on this fixed
three-scene subset.

## Terminal counterfactual

The fail-closed materializer follows each association through all strictly
later native aliases, validates the terminal row mapping, and changes only a
case where candidate and target survive as distinct terminal outputs.

| terminal class | count |
|---|---:|
| later native association made them identical | 0 |
| candidate later dropped | 1 |
| target later dropped | 2 |
| both survive as distinct outputs | 1 |

The one materializable case is `scene0568_00`, frame 1400: candidate track 431
later aliases to terminal track 387, while target track 368 survives
independently.  The create-only counterfactual removes terminal row 20 (track
387), reducing the fixed-subset output from 46 to 45 boxes while preserving
every retained row's order, class, score and geometry exactly.

| arm | boxes | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| T05 + Graw-shadow native output | 46 | 26.8814 | 26.8814 | 18.1330 |
| Graw create-only counterfactual | 45 | 24.3208 | 24.3208 | 18.4654 |
| counterfactual - native | -1 | **-2.5606** | **-2.5606** | **+0.3324** |

At IoU 0.15 and 0.25, recall falls from 64.2857% to 60.7143%, showing that
the deleted candidate contributes a true positive under those thresholds.
The AP50 increase does not compensate for the large low-IoU recall loss.

## Gate decision

`Graw-shadow` is complete and passes its systems checks.  Raw `Graw-active`
fails the preregistered accuracy gate and must remain disabled.  This result is
only a three-scene gate, not a full100 accuracy estimate, but its negative
direction is sufficient to prohibit spending a full100 run on the unchanged
raw matcher.

The next route stage is `SMOV-shadow`, followed by `Gclean-active` only if
cleaned fragments remove these false raw overlaps and pass the same
counterfactual gate.

Artifacts:

- identity comparison: `logs/scannet_graw_shadow_identity_score05.json`
- counterfactual audit:
  `results/scannet_graw_counterfactual_score05_preflight3_v2/graw_counterfactual_audit.json`
- machine-readable result: `logs/scannet_graw_shadow_preflight_score05.json`

