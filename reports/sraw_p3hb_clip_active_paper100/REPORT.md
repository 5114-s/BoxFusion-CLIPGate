# SRAW-P3HB-CLIP-v1 active paper100 result

Date: 2026-08-30

## Decision

Reject this active birth policy and do not accumulate it into Cbest. It is
causal and target-data-training-free, but its admission funnel is too strict:
the only shadow birth is removed by the frozen terminal native-overlap gate,
so the active output is exactly the unchanged Cbest prediction set.

## Protocol

- ScanNet official ordered paper100 scene list.
- Native prefix: Cbest (`Reliable-View Top-K + Boxer active`), CuTR
  `score_thresh=0.5`, appearance gate disabled.
- Evaluation: official evaluator SHA-256
  `aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27`;
  every final score is forced to `1.0`.
- Frozen route: F4/SRAW sources -> first three distinct past-only views ->
  HB source medoid -> frozen geometry gate -> unchanged native 473-way CLIP
  gate -> causal score-0.5 CuTR proposal-history novelty -> past-birth NMS and
  cap 2 -> terminal Cbest reconciliation.
- No ScanNet annotation, GT, evaluator result, fine-tuning, optimizer, or
  online learning is available to the selector. OWLv2, BoxerNet and OpenCLIP
  remain externally pretrained and frozen.

## Candidate funnel

| Stage | Count |
|---|---:|
| L2 tracks | 28,156 |
| First-three-view eligible tracks | 5,548 |
| Geometry-pass tracks | 222 |
| CLIP crops | 666 |
| Semantic-gate rejects | 210 |
| Causal CuTR-overlap rejects | 11 |
| Shadow births | 1 |
| Terminal native-overlap rejects | 1 |
| Final appended births | 0 |

The shadow birth is `scene0088_03/track3`, confirmed at frame 50 from frames
0/25/50. Terminal reconciliation rejects it as overlapping an existing Cbest
prediction.

## Official paper100 AP

All numbers are AP points.

| Output | Boxes | AP15 | AP25 | AP50 |
|---|---:|---:|---:|---:|
| Cbest reference | 1,788 | 31.0130 | 26.7911 | 12.0669 |
| Cbest + SRAW-P3HB-CLIP-v1 | 1,788 | 31.0130 | 26.7911 | 12.0669 |
| Delta | 0 | **0.0000** | **0.0000** | **0.0000** |

A strict two-root comparison verified identical scene sets, row counts,
classes, scores, row order, and all 1,788 corresponding box corners with zero
geometric error.

## Interpretation

The earlier SRAW AP50 oracle gain (`+10.5851`) is not a deployable result: it
uses GT to select source identities and geometry. This experiment replaces
that oracle with one frozen GT-free policy. The result shows that this policy
cannot realize the oracle capacity; it does not show that the raw proposal
pool itself has no useful boxes.

This result also does not invalidate every previous module. Same-protocol
experiments already found positive effects for Reliable-View Top-K, Boxer
active geometry replacement, and the much smaller CLIP-vocab birth-v3. Other
shadow-only stages did not mutate predictions and therefore had no AP effect
to measure, while birth-v2 and UDC/MobileSAM produced negative active deltas.

## Evidence

- Frozen policy: `docs/SRAW_P3HB_CLIP_V1_PROTOCOL_FREEZE.md`
- Shadow ledger:
  `logs/scannet_sraw_p3hb_clip_shadow_paper100_score05/SRAW_P3HB_CLIP_SHADOW_PAPER100.json`
- Active manifest:
  `results/scannet_sraw_p3hb_clip_birth_score05/SRAW_P3HB_CLIP_BIRTH_PAPER100.json`
- Official evaluation:
  `logs/cgf_paper100_constant_score/scannet_sraw_p3hb_clip_birth_score05_constant.log`
- Focused tests: 40 passed across the shadow runner and materializer suites.

The AP run is a sealed replay-based active-output test. It does not by itself
prove full end-to-end live FPS for the added branch.
