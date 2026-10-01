# Reliability-v2 isolated development line

This tree is intentionally separate from the active BoxFusion workspace.  It
must not be used to overwrite the frozen v4 baseline while official100 jobs
are running.

## Implemented components

- `boxfusion/causal_reliability.py`: bounded causal, direction-diverse soft
  evidence state.  Its lower bound is beta-inspired and is not claimed to be
  a calibrated posterior.
- `boxfusion/native_reliability_reranker.py`: MVSR-v2.  It combines proposal
  quality, visibility, 2D overlap and 3D consistency, resets stale evidence
  after a large native-geometry change, and changes native scores only.
- `boxfusion/m1_anchor_reliability_online.py`: CALR-v2.  The voxel grid is a
  spatial index; association uses center, size and IoU gates across nearby
  cells.  Confirmation requires directionally diverse causal support, and
  outputs form a revisable bounded top-K shadow set.
- `boxfusion/reliability_candidate_map.py`: full online composition with the
  frozen proposal-only PLR.

The runnable config is
`config/scannet_t05_boxer_reliability_v2_official100.yaml`.  It writes only
under this development tree.

## Evaluation gates

1. Replay MVSR operators with identical native geometry and box count:
   native, first, mean, max, EMA, diverse-max, and reliability.
   These v2 operators use the same composite evidence strength; the frozen
   v4 raw-2D-IoU running-max route remains a separate control.
2. Promote MVSR-v2 only if it improves that frozen running-max control under
   the same cached inputs and protocol.
3. Compare CALR-v2 with the frozen exact-voxel CALR at equal output budget and
   fixed false-positive budgets.  Do not tune on official100 test outputs.
4. Run the matched screening pass only after the frozen v4 official100 job
   finishes; defer cross-dataset and final efficiency runs until selection.

The current execution order is stricter: the old automatic CA-1M/FPS
follow-up is paused. After v4 finishes, one screening run exports all MVSR
aggregation arms plus frozen-CALR, CALR-v2, and frozen-PLR components from the
same online pass. It then stops with selection pending. CA-1M, matched FPS,
PLR-v2 development, extra50, and paper-table runs begin only after the
MVSR/CALR screening result is reviewed and the accepted pair is frozen.

No cross-mapper generality or calibrated-probability claim follows from this
implementation alone.
