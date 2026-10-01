# MV3DIS-Depth-Lite S0 (shadow only)

## Scope

This module borrows only the training-free depth consistency and visibility
idea from MV3DIS. It is a bounded causal observer for the existing BoxFusion
stream; it is not the full offline MV3DIS pipeline.

The S0 implementation does not change native association, fusion, geometry,
scores, labels, CLIP calls, or exported boxes. Consequently its current AP
gain is exactly zero. Its purpose is to measure whether cross-view depth
evidence can safely veto high-confidence false births before any active rule is
authorized.

## Causal data flow

For each real CuTR keyframe:

1. Moon-QIM-lite queries only the previously committed track index.
2. A sparse guide of at most 64 RGB-D points is sampled for each current
   proposal from the intersection of its raw 2D box and projected 3D OBB.
3. For each active QIM Top-3 candidate, up to five historical sparse guides are
   projected into the current RGB-D frame and current raw proposal box.
4. Native BoxFusion association runs unchanged.
5. Only after native association, the current sparse guides are committed to
   the resolved stable track IDs. The module never stores historical depth
   images.

The terminal-frame branch that reuses stale CuTR proposals never queries or
commits this observer.

## Frozen S0 evidence

For historical guide point `j`, MV3DIS-style visibility and depth consistency
are

```
I_vis(j) = inside_image(j) * 1(|z_j - d_j| < alpha * d_j)
w_d(j)   = 1 - |z_j - d_j| / (alpha * d_j)
```

with `alpha=0.05`. The current-frame visibility is

```
V_f = sum(I_vis) / |guide|
```

and the proposal-conditioned backward evidence is

```
V_b = sum(I_vis * inside_current_box) / max(sum(I_vis), 1)
D_b = sum(I_vis * inside_current_box * w_d)
      / max(sum(I_vis * inside_current_box), 1)
a   = V_b * D_b
```

A historical view supports a candidate only when `V_f > 0.30` and
`V_b > 0.90`. A row is marked `would_veto_birth` only when exactly one QIM
candidate has at least two supporting committed views, all attempted
projections are complete, and its affinity share is greater than `0.90`.
Even then the emitted action remains `defer_to_native` in S0.

The same-frame guide self-projection quality is diagnostic only. It is not a
BoxFusion view weight, so summaries explicitly report
`fusion_weights_computed=false` and `fusion_weights_applied=false`.

## Safety and resource bounds

- no training, learned parameters, ground truth, detector scores, CLIP, PUF,
  or online parameter updates;
- at most 256 proposals, three QIM candidates, five guides per track, 64 points
  per guide, and 8,192 projected points per branch per keyframe;
- at most 80 committed frame IDs are retained; full RGB-D frames are not kept;
- malformed sensors, missing guides, budget exhaustion, ambiguous committed
  IDs, or incomplete projections abstain and defer to native BoxFusion;
- thresholds are frozen in code when the observer is enabled.

Use
`config/scannet_qim_puf_arbitration_mv3dis_shadow.yaml` for the real-stream
shadow run. The log contains one machine-readable line prefixed by
`MV3DIS-Depth-lite S0 shadow JSON | `.

## Activation policy

S0 is not active-eligible. First run a real, paired shadow evaluation and
measure:

- whether the pre-registered false-birth row
  `(scene0598_01, frame 175, proposal 18)` contains native-relative track 15 in
  QIM Top-3 and whether two-view depth evidence recommends it;
- veto precision and coverage across a frozen scene set, not only that row;
- numerical/output identity and end-to-end throughput, including guide
  extraction and projection.

Only a separate counterfactual active experiment, with thresholds still
frozen and validation ground truth used solely for final evaluation, can
establish an AP improvement. No percentage gain is claimed by S0.

## Real-stream S0 result (scene0598_01)

The frozen implementation was run once against a same-code control on the
pre-registered regression scene:

- 99 proposals over 32 real CuTR keyframes;
- 23 shadow veto recommendations, 19 native-relative correct and 4 wrong
  (`82.61%` precision; `29.23%` correct-veto coverage of native history);
- the known row `(175, 18)` retrieved track 15 in QIM rank 0, but that track
  had zero retained historical guides, so S0 correctly abstained and did not
  repair the PUF false-birth recommendation;
- observer/control FPS was `31.25/31.82 = 0.9821`; measured observer wrapper
  overhead was `0.13094 ms` per input frame;
- exported row count, labels, and scores were unchanged. CUDA geometry differed
  by at most `3.8147e-5 m`, within the registered numerical identity tolerance,
  but the pickle was not byte-identical.

The paired report is
`reports/mv3dis_depth_lite/scene0598_s0_v1.json`. These results reject active
use of the S0 veto rule. They do not show an AP improvement; because S0 never
changes predictions, its measured AP delta is zero by construction.
