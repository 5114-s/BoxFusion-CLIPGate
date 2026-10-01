# Reliability-v2 screening protocol

This protocol is fixed before the reliability-v2 official100 screening run.
It selects engineering variants for later validation; it does not turn the
ScanNet validation split into a claim of cross-dataset generality.

## MVSR gate

All arms must keep native geometry, labels and row count fixed. The frozen v4
raw-2D-IoU running maximum is the control. Among first, mean, max, EMA,
diverse-max and reliability, rank variants by AP25, then AP50, then AP15.
Promote a v2 arm only when:

- AP25 improves by at least 0.20 point over frozen v4 raw-max;
- neither AP15 nor AP50 decreases by more than 0.10 point;
- the same arm is not dominated at all three IoU thresholds by another v2 arm.

Otherwise retain frozen v4 MVSR. No threshold or update weight is retuned from
the official100 result.

## CALR gate

Compare frozen exact-voxel CALR and CALR-v2 under the selected MVSR arm. Report
both full outputs and per-scene matched output budgets. CALR-v2 is promoted as
a precision-oriented recovery branch only when matched-budget recall at IoU
0.25 improves at one or more of 5, 10 or 20 FP per 100 GT, while AP25 and AP50
do not fall by more than 0.10 point. If only long-tail AP improves, retain the
method but describe it as long-tail recovery rather than high-precision
recovery. If neither criterion holds, retain frozen CALR.

## PLR gate

Only after the MVSR/CALR pair is selected and hashed, compare frozen PLR-v1,
`+Assoc`, `+Reliability`, and `+Score` sequentially. All variants use the same
post-NMS proposals, no child evidence, maximum 12 outputs per scene, direct
branch concatenation, and identical selected MVSR/CALR states. A later PLR
stage is retained only if it improves the strong-prefix AP25 or fixed-FP
recall without reducing AP50 by more than 0.10 point. Actual output counts,
PLR-only GT coverage, CALR overlap, duplicate rate, latency and output-set
stability are reported for every retained stage.

CA-1M, extra50, matched FPS and final paper tables remain paused until these
gates finish. Those later datasets are validation targets, not tuning inputs.
