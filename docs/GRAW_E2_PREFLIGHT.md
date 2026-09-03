# Graw E2 proposal-replay preflight

Date: 2026-08-23 (Asia/Shanghai)

This is a prerequisite identity experiment.  Neither Graw-shadow nor
Graw-active is enabled in these three arms.

## Frozen protocol

- Base arm: T05 (`score_thresh=0.5`, appearance gate disabled,
  Reliable-View Top-K enabled).
- Ordered scenes: `scene0568_00`, `scene0606_01`, `scene0377_02`.
- The scenes cover a normal sequence, a high-proposal sequence, and the
  released frame-0 retry path.
- Cache namespace:
  `scannet-graw-e2-score05-preflight3-v3-r1`.
- Producer fingerprint:
  `457c997631cd71a83b6480a6e45e103e273ac5ed2d1488252790549c2e2b3504`.

The sealed index contains 1,163 proposals in 209 keyframe records:

| Scene | keyframe records | proposals | final boxes |
|---|---:|---:|---:|
| scene0568_00 | 66 | 489 | 22 |
| scene0606_01 | 113 | 580 | 19 |
| scene0377_02 | 30 | 94 | 5 |

`scene0377_02/frame_000000` is sealed with `attempt_id=retry`; all later
records use `primary`.

## Identity result

E2-record, E2-replay-1 and E2-replay-2 have exactly the same ordered scene
set, final row counts, class values, score values and score order.  Proposal
replay itself checks every cached keyframe's input signature, attempt ID,
count/order, protected-field hashes, geometry hash and RNG hash.

The measured cross-process terminal-geometry floor is:

| Pair | box-error p50 | box-error p95 | maximum |
|---|---:|---:|---:|
| record vs replay-1 | 0.000 mm | 0.217 mm | 14.277 mm |
| record vs replay-2 | 0.000 mm | 0.038 mm | 14.296 mm |
| replay-1 vs replay-2 | 0.000 mm | 0.139 mm | 0.245 mm |

This passes the preregistered global safety caps of p95 <= 2 mm and maximum
<= 15 mm.  The large record-vs-replay maximum is one stochastic BoxFusion
corner in `scene0568_00`; it is not proposal-cache drift.

## Three-scene AP threshold check

All three arms produce identical subset metrics:

| arm | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| E2-record | 26.8814 | 26.8814 | 18.1330 |
| E2-replay-1 | 26.8814 | 26.8814 | 18.1330 |
| E2-replay-2 | 26.8814 | 26.8814 | 18.1330 |

These are a three-scene identity check, not an accuracy estimate and not
comparable to full100 AP.

Machine-readable geometry comparison:
`logs/scannet_graw_e2_replay_floor_score05.json`.

## Gate decision

The sealed replay prerequisite passes.  The next allowed experiment is
registry-only/Graw-shadow.  Graw-active remains prohibited until shadow proves
noninterference, bounded latency, enough eligible unmatched proposals, and a
credible counterfactual TP/FP signal.
