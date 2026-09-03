# Boxer-Past3 S1 receipt + depth shadow preregistration

Date: 2026-08-23

## Question and status

Can a frozen universal proposal branch recover missed ScanNet objects without
training by requiring a causal, depth-supported, materially different
three-view confirmation?

S1 is shadow-only.  It is not authorized to append a prediction.  The method
and thresholds below are fixed before opening any per-scene H10 ground truth.
The three S0 scenes have already been inspected with GT and remain
development-only.

## Native and frozen inputs

- Native prefix: completed T05, `score_thresh=0.5`, appearance gate disabled,
  Reliable-View Top-K3, frozen CuTR and frozen CLIP.
- Formal score mode: constant `1.0` for every native or appended prediction.
- Proposal source: frozen OWLv2 + Boxer profile specified by
  `BOXER_PAST3_S1_H10_PROPOSAL_CONTRACT.md`.
- Online order: sealed gap-25 T05 schedule.  Missing/invalid-pose frames are
  omitted and never replaced with future frames; zero-candidate scheduled
  frames still advance tracker time.
- No detector label, CLIP feature, ScanNet annotation, oracle output, learned
  calibrator, fitting, gradient update, or target-dataset checkpoint is used.

## Immutable geometry receipt

Each scheduled keyframe is processed by an explicit `query` then `commit`.
The query sees only state committed by earlier keyframes, so observations from
one frame cannot confirm each other.  The transferred S0 geometry tracker uses:

- at most 64 observations per frame, 1,024 live tracks, 5 stored observations
  per track, and a 10-keyframe TTL;
- within-frame AABB deduplication at IoU `0.50`;
- association only when AABB IoU is at least `0.10` **and** center distance is
  at most `0.50 m`;
- at least 3 distinct frames, median pairwise AABB IoU at least `0.25`, center
  RMS at most `0.25 m`, and medoid minimum AABB extent at least `0.30 m`.

At the first keyframe satisfying these conditions, the OBB medoid, receipt
frame, and three-to-five evidence row/frame IDs are copied and made immutable.
Later matching observations may provide depth evidence, but cannot change the
receipt OBB or its original provenance.

## Fixed causal depth/view graph

Every scheduled keyframe, including a zero-proposal keyframe, first advances a
FIFO RGB-D/pose ring containing only the current keyframe and the preceding 10
keyframes.  Its hard capacity is therefore 11, matching the track TTL plus the
current frame.  A historical lookup may read only a still-resident ring entry;
it cannot reload an evicted frame from disk or access a future/off-schedule
frame.  Missing or evicted evidence abstains.

The graph holds at most the 5 most recent causal evidence nodes associated
with an immutable receipt.  At receipt time its original evidence is replayed
only from that 11-keyframe ring.  A node is valid only when the resident depth
image, depth intrinsics, and finite camera pose yield at least 16 and at most
64 guide points inside the fixed receipt OBB.  After a node/edge has been
processed, only its bounded geometry guide or scalar metrics are retained;
raw historical RGB-D is not copied into per-object state.

For a chronological historical-to-current node pair:

- depth tolerance `alpha = 0.05`;
- forward visibility `Vf > 0.30` and backward consistency `Vb > 0.90`;
- camera-center baseline at least `0.15 m`; and
- object-centered viewing-ray angle at least `10 degrees`.

`Df`, `Db`, and any aggregate affinity are diagnostics only and have no
threshold.  A receipt becomes depth-qualified at the first keyframe for which
one weakly connected component contains at least 3 distinct-frame nodes and at
least 2 supporting edges.  Qualification time and graph evidence are then
frozen.  There is no look-ahead and no terminal backfill.

These numbers are direct transfers of the existing R1 depth rule and the S0
view-diversity rule.  A no-GT replay on S0 was used only to verify feasibility
and latency, not to select thresholds.

## Terminal output-inert filter

Only depth-qualified immutable receipts enter the transferred S0 terminal
filter:

- native novelty: maximum terminal T05 AABB IoU below `0.10`;
- side-candidate NMS AABB IoU `0.25`;
- at most 6 candidates per scene;
- frozen detector probability is used only for deterministic bounded ranking.

Native T05 predictions are read only after causal qualification for terminal
duplicate suppression.  Their class, geometry, score, order, CLIP vocabulary,
and embeddings must remain byte-identical.  The sealed S1 sidecar records
`birth=false`, `active_authorized=false`, and
`native_mutation_applied=false`.

## Scene splits and decision sequence

1. Development sanity check: run the fixed S1 method on the already-open S0
   three scenes.  If the fixed suffix adds no maximum-matching TP at any one of
   IoU 0.15/0.25/0.50, or reduces AP at any threshold, reject S1 without
   opening H10 GT.
2. One-shot H10 gate: only after the development sanity check passes, seal the
   shadow sidecar for the 10 predeclared scenes in
   `scannetv2_boxer_past3_s1_holdout10.txt`, then open GT once.  Promotion
   requires at least one new maximum-matching TP at all three IoUs, fixed-suffix
   AP delta nonnegative at all three IoUs, no native hash change, and no audit
   incompleteness/cap overflow.
3. Confirmatory C87: if H10 passes, freeze code and artifacts unchanged and run
   once on `scannetv2_boxer_past3_s1_confirm87.txt`.  C87, not H10, is the
   primary untouched confirmation set.
4. Official full100 is reported only as a secondary/descriptive aggregate
   because the S0 and H10 scene outcomes have then been inspected.

H10 list SHA-256:
`8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90`.
C87 list SHA-256:
`3fb0f8bc79217cfe3ce47bf05970b3a4f75981e357a50a7804cf51f0e4c77b2c`.

Passing these gates establishes that the branch is useful, not that it has
already achieved the requested +10 absolute AP points.  A +10 claim requires
the unchanged C87/full100 constant-score evaluation and paired uncertainty
analysis.
