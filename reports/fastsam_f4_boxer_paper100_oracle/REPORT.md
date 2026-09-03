# F4 FastSAM/F2 + frozen BoxerNet geometry shadow — paper100

Date: 2026-08-29 (Asia/Shanghai)

Protocol: `F4-FASTSAM-BOXER-GEOMETRY-SHADOW-PAPER100`

## Outcome

F4 passed the preregistered integrity, causality, steady-state runtime,
geometry-capacity, and constructive-oracle gates.  The result authorizes only
a new preregistered GT-free selector experiment.  It does **not** authorize
active birth and is not a deployable AP result.

All 100 scenes, 6,817 scheduled keyframes, 6,726 successful frames, and
52,299 sealed FastSAM/F2 source identities were verified.  BoxerNet produced
52,299 valid HB hypotheses and zero abstentions.  Native predictions, score
1.0, CLIP, labels, ordering, and outputs were unchanged.

## Constant-score geometry oracle

The table reports the native constant-score baseline, the historical F2
geometry group `Gbase={H0,HL,HLG}`, and the new source-identity-constrained
group `G4={H0,HL,HLG,HB}`.  Suffix rows are selected with GT and are therefore
oracle-only.

| IoU | Native AP | Gbase oracle AP | G4 oracle AP | G4 - native | G4 - Gbase |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.15 | 31.0130 | 51.4312 | 54.9690 | +23.9560 | +3.5378 |
| 0.25 | 26.7911 | 41.0233 | 48.4223 | +21.6312 | +7.3990 |
| 0.50 | 12.0669 | 15.0828 | 22.6520 | +10.5851 | +7.5692 |

HB-only oracle deltas over native were `+20.7521 / +18.6058 / +9.3346`
at IoU `0.15 / 0.25 / 0.50`.  Thus HB alone narrowly misses +10 at IoU 0.50,
while the fixed alternative group G4 crosses +10 at all three thresholds.

## Matching capacity

| IoU | Gbase additional native-union matches | G4 additional matches | HB marginal gain over Gbase |
| --- | ---: | ---: | ---: |
| 0.15 | 399 | 473 | +74 |
| 0.25 | 282 | 421 | +139 |
| 0.50 | 74 | 260 | +186 |

The high-IoU result is the main positive signal: HB adds 186 source-identity
constrained matches beyond the complete F2 geometry group at IoU 0.50.

## Reproduction and runtime

- Native AP was reproduced as `31.0130259031 / 26.7911284298 /
  12.0668518301` within numerical tolerance.
- H0 reproduced F1 exactly, and H0/HL/HLG/Gbase reproduced the sealed F2
  oracle exactly, including F2 additional matches `399 / 282 / 74`.
- F4 incremental warm p95: `65.453 ms/keyframe` (gate: <=100 ms).
- Replay-composed warm p95/max: `287.055 / 450.867 ms` (gates: <=350 / <833.33 ms).
- Replay-composed mean divided by the gap-25 source-frame stride:
  `8.397 ms` (gate: <=14 ms).
- Warm deadline misses: `0`; two cold-start misses remain explicitly recorded.
- Recorded replay peak CUDA memory: `632,576,000 bytes` (gate: <=4 GiB).

These replay gates pass, but promotion still requires the separately frozen
same-GPU live Cbest + FastSAM + Boxer benchmark at >=15 FPS.

## Decision

`f4_pass_authorize_new_preregistered_gt_free_selector_only`

Next experiment: preregister a past/current-only, GT-free hypothesis selector
for H0/HL/HLG/HB.  Run it first in shadow on the sealed paper100 outputs.  Only
if that selector has sufficient precision should an active low-score birth
paper100 evaluation and the live >=15 FPS benchmark be run.

Machine-readable results:

- `F4_FASTSAM_BOXER_PAPER100_ORACLE.json`
- `../../logs/scannet_fastsam_f4_boxer_paper100_score05/final/F4_FASTSAM_BOXER_PAPER100.json`
