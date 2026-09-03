# UDC + frozen MobileSAM v1.1 — official100 result

Date: 2026-08-25

## Decision

Reject this active birth module and do not accumulate it into Cbest.  It adds
two true positives at IoU 0.15/0.25, but the other 27 appended boxes reduce AP
at every evaluation threshold.  It is far below the requested +10 AP-point
target.

## Protocol

- Dataset/split: ScanNet official100, all 100 scenes.
- Frozen input schedule: 6,817 gap-25 RGB-D keyframes and 23,651 cached CuTR
  boxes at `score_thresh=0.5`.
- Reference prefix: T05 + Boxer active (Cbest), 1,788 boxes.
- Evaluation: official constant-score protocol; every output score is `1.0`.
- Learning: no ScanNet fitting, calibration, fine-tuning, or online update.
  MobileSAM is a frozen generally pretrained model (10,130,092 parameters;
  checkpoint SHA-256 `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`).
- Causality boundary: UDC generation and three-view confirmation are causal
  and past-only.  Terminal deduplication reads the final native prefix, so this
  replay is not proof of fully integrated strict-online native novelty.

## Full100 AP

All values below are AP points, not fractions.

| Result | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| Cbest reference | 31.0130 | 26.7911 | 12.0669 |
| Cbest + UDC/MobileSAM v1.1 | 30.7656 | 26.5374 | 11.9059 |
| Delta attributable to UDC | **-0.2474** | **-0.2537** | **-0.1610** |

The active output contains 1,817 boxes: the exact 1,788-row native prefix plus
29 appended births across 23 scenes.  The native rows were verified unchanged
at the category, corner dtype/shape/bytes, and score-value level.

## Capacity and GT-after-the-fact diagnosis

The no-GT shadow produced 54,708 raw components, 10,163 eligible components,
6,632 MobileSAM prompts, 4,581 accepted mask lifts, 305 first-three-view
confirmations, and 43 pre-novelty receipts in 28 scenes.  This failed the
preregistered capacity reference of 300 confirmations, 250 candidates, and 80
candidate scenes.

Terminal native-containment filtering rejected 14 receipts and appended 29.
After evaluation, a diagnostic comparison against all same-scene GT boxes
showed:

| Birth max IoU to any GT | Count |
|---|---:|
| exactly 0 | 7 |
| `(0, 0.15)` | 20 |
| `[0.15, 0.25)` | 0 |
| `[0.25, 0.50)` | 2 |
| `[0.50, 1]` | 0 |

The two useful boxes recover two previously unmatched GT instances in
`scene0081_02`, raising TP from 869 to 871 at IoU 0.15 and from 809 to 811 at
IoU 0.25.  This exactly explains the recall increase of
`2 / 1433 = 0.00139567`.  Marginal birth precision is therefore only 2/29
(6.90%) at IoU 0.15 and 0.25, and 0/29 at IoU 0.50.

## Runtime evidence

Measured on an RTX 3090:

| Incremental branch timing | Mean | p50 | p95 |
|---|---:|---:|---:|
| UDC preprocessing | 4.110 ms | 3.798 ms | 6.590 ms |
| MobileSAM + lifting, prompted frames | 52.885 ms | 44.953 ms | 110.573 ms |
| Complete branch per keyframe | 56.359 ms | 55.255 ms | 123.731 ms |

At the gap-25 schedule, measured branch overhead amortizes to 2.254 ms per raw
stream frame.  Peak GPU allocation was 305,778,176 bytes.  These are branch
measurements, not a proof of end-to-end BoxFusion FPS.

## Verification and evidence

- Targeted unit/contract tests: 39 passed.
- Shadow artifact: `logs/scannet_udc_mobilesam_full100_edgefix_v11_score05/UDC_MOBILESAM_FULL100.json`.
- Active manifest: `results/scannet_udc_mobilesam_birth_edgefix_v11_score05/UDC_MOBILESAM_BIRTH_FULL100.json`.
- Active evaluation: `logs/cgf_paper100_constant_score/scannet_udc_mobilesam_birth_edgefix_v11_score05_constant.log`.
- Reference evaluation: `logs/cgf_paper100_constant_score/scannet_t05_boxer_replay_active_score05_constant.log`.
