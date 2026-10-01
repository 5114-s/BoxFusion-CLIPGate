# Boxer-Past3 S0 shadow preregistration

Date: 2026-08-23

## Question

Can the already sealed, frozen OWLv2 + Boxer per-view proposal pool be reduced
to a small set of novel, geometrically stable object candidates by a causal,
training-free, past-only three-view confirmer?

This is an output-inert shadow experiment.  It is not authorized to append a
box.  Ground truth may be opened only by the separate post-hoc oracle after the
shadow sidecar and its hashes have been sealed.

## Frozen inputs

- Native arm: T05 (`score_thresh=0.5`, appearance gate disabled,
  Reliable-View Top-K3, frozen CuTR and CLIP).
- Formal evaluator score mode: constant `1.0` for every prediction.
- Boxer input: `boxfusion.owl_boxer_shadow_candidates.v1`, generated with the
  frozen OWLv2 1,220-prompt LVIS+ detector and frozen BoxerNet.
- Development/preflight scenes: `scene0568_00`, `scene0606_01`,
  `scene0377_02` in their sealed online keyframe order.
- The shadow confirmer reads neither ScanNet annotations nor CLIP embeddings.

## One-shot transferred confirmer

The association and terminal stability policy is transferred without fitting
from `tools/boxfusion_tr3d_pipeline/boxfusion/cutr_residual_birth_lite.py`:

- per-frame cap: 64 observations;
- within-frame AABB deduplication IoU: 0.50;
- association: AABB IoU at least 0.10 or center distance at most 0.50 m;
- confirmation: at least three distinct causal keyframes;
- track TTL: 10 keyframes; at most 5 stored observations;
- terminal median pairwise AABB IoU at least 0.25;
- terminal center RMS at most 0.25 m;
- every AABB extent at least 0.30 m;
- native novelty: maximum T05 AABB IoU below 0.10;
- side-candidate NMS IoU: 0.25;
- at most 6 candidates per scene.

The raw frozen detector probability is used only for deterministic observation
and candidate ranking.  No score is learned, calibrated, or written back to a
native prediction.

To ensure that three timestamps are also materially different viewpoints, an
S0 candidate additionally needs both:

- maximum camera-center baseline at least 0.15 m; and
- maximum viewing-ray angular span at least 10 degrees.

These constants are frozen before inspecting the S0 candidate oracle.  The
current hard unexplained-depth ratio is not used as an admission gate; it was
already rejected at AP50 and may later be reported only as a soft diagnostic.

## Invariants

- Frames are consumed in strictly increasing sealed schedule order.
- An observation can match only state committed by earlier/current frames;
  future frames are never substituted for an invalid pose.
- Candidate geometry is the fixed AABB-IoU medoid of its bounded observations.
- Native class, corner, score and row order are byte-identical before and after
  shadow materialization.
- Native CLIP category, vocabulary and embedding are not read or changed.
- The sidecar records `birth=false`, `active_authorized=false`, and
  `native_mutation_applied=false`.

## Preflight decision rule

After the shadow sidecar is sealed, a separate GT-only oracle will report:

1. candidate precision/coverage at IoU 0.15, 0.25 and 0.50;
2. recovery of GT missed by the official constant-score T05 prefix;
3. constant-score fixed-suffix AP and false positives; and
4. CPU confirmer latency plus the already measured frozen proposal latency.

Active birth is not enabled automatically.  S0 is promoted to an active
three-scene counterfactual only if it adds at least one new maximum-matching GT
at every IoU threshold, does not reduce constant-score AP at any threshold,
and keeps every native prediction hash unchanged.  A full100 run requires a
separate promotion decision and an integrated same-GPU realtime measurement.
