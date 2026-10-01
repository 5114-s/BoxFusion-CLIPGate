# SMOV-shadow and Gclean preflight

Date: 2026-08-23 (Asia/Shanghai)

## Frozen protocol

- Base arm: T05 (`score_thresh=0.5`, native scores preserved, appearance gate
  disabled, Reliable-View Top-K enabled).
- Ordered scenes: `scene0568_00`, `scene0606_01`, `scene0377_02`.
- Both arms replay the same sealed 1,163 CuTR proposals from
  `scannet-graw-e2-score05-preflight3-v3-r1`.
- No module is trained or fitted on ScanNet.  SMOV uses fixed depth rules and
  Gclean uses the frozen Group3D-lite thresholds.
- Both experiments are causal, online and output-inert.  Gclean queries only
  begin-frame-past tracks not already reserved by native association and
  commits current observations only after the query.

SMOV uses a 0.15 m full-resolution depth-edge threshold, a 0.15 m
center-seeded component jump threshold, and direct signed 5 cm voxel keys
`floor(world / 0.05)`.  The matcher never re-quantizes float centroids.

## SMOV-shadow

All three traces are valid, with no observer errors or proposal cap events.

| quantity | result |
|---|---:|
| nonempty keyframes | 199 |
| sealed proposals | 1,163 |
| accepted clean fragments | 1,027 (88.31%) |
| abstained fragments | 136 |
| proposal-cap abstentions | 0 |
| center seed unusable | 89 |
| insufficient points after voxelization | 35 |
| insufficient component pixels | 12 |
| prepare latency p50 | 21.446 ms/keyframe |
| prepare latency p95 | 48.860 ms/keyframe |
| prepare latency maximum | 194.114 ms/keyframe |

At `gap=25`, the observed p95 and maximum amortize to approximately 1.95 ms
and 7.76 ms per input frame.  The three scene wall times are 70.23 s, 92.27 s
and 22.55 s, respectively.

Scene/class/score/row order and all 46 terminal rows are preserved exactly.
One stochastic BoxFusion corner in the standalone SMOV process differs by
17.70 mm from an E2 replay, slightly above the earlier 15 mm empirical replay
maximum; global p95 remains 0.244 mm and the three AP values are unchanged.
The Gclean process below, which includes the same SMOV extraction, has a
14.32 mm maximum and is inside the preregistered replay cap.

## Gclean-shadow

SMOV cleaning reduces eligible native-unmatched candidates from Graw's 218 to
199.  Gclean records four associations and no matcher fail-open event:

| scene / frame | candidate -> past | centroid distance | Dice | Jaccard | terminal class |
|---|---|---:|---:|---:|---|
| scene0568_00 / 50 | 7 -> 3 | 0.134 m | 0.5625 | 0.3913 | target dropped |
| scene0568_00 / 375 | 124 -> 19 | 0.040 m | 0.4974 | 0.3310 | target dropped |
| scene0606_01 / 325 | 62 -> 56 | 0.171 m | 0.5079 | 0.3404 | candidate dropped |
| scene0606_01 / 1700 | 360 -> 292 | 0.485 m | 0.3287 | 0.1967 | later native same |

The `centroid_distance` stored in matcher diagnostics is measured in 5 cm
voxel units; the table converts it to metres.

SMOV removes the raw Graw frame-1400 `both-survive-distinct` association that
caused the AP15/AP25 loss.  The new terminal classification is:

| terminal class | Graw raw | Gclean |
|---|---:|---:|
| later native same | 0 | 1 |
| candidate dropped | 1 | 1 |
| target dropped | 2 | 2 |
| both survive distinct | 1 | **0** |

Gclean total observer latency is 22.280 ms/keyframe at p50, 52.280 ms at p95
and 204.863 ms maximum.  At `gap=25`, p95 amortizes to about 2.09 ms per input
frame.  Its terminal output preserves exact class, score and row order; the
maximum control geometry difference is 14.317 mm and global p95 is 0.091 mm.

## Counterfactual AP

The strict terminal materializer finds no `both-survive-distinct` association,
so it deletes zero rows.  Native and counterfactual roots have byte-identical
prediction files for all three scenes.

| arm | boxes | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| T05 + Gclean-shadow native output | 46 | 26.8814 | 26.8814 | 18.1330 |
| Gclean create-only counterfactual | 46 | 26.8814 | 26.8814 | 18.1330 |
| counterfactual - native | 0 | **+0.0000** | **+0.0000** | **+0.0000** |

## Gate decision

SMOV cleaning is retained as a safe observer component because it removes the
known harmful raw overlap and stays within the online budget.  Unchanged
Gclean association, however, has zero terminal-actionable matches and zero AP
gain on the fixed subset.  `Gclean-active` therefore remains disabled and no
full100 run is justified.  This stage cannot approach a +10 AP objective
because it neither creates proposals nor changes recall.

The next planned gate is `PUF-shadow`, applied only to Gclean's secondary
association probabilities and without enabling birth.

Artifacts:

- SMOV identity: `logs/scannet_smov_shadow_identity_score05.json`
- Gclean identity: `logs/scannet_gclean_shadow_identity_score05.json`
- exact native/counterfactual comparison:
  `logs/scannet_gclean_counterfactual_identity_score05.json`
- Gclean counterfactual audit:
  `results/scannet_gclean_counterfactual_score05_preflight3/gclean_counterfactual_audit.json`
- machine-readable summary:
  `logs/scannet_smov_gclean_preflight_score05.json`

