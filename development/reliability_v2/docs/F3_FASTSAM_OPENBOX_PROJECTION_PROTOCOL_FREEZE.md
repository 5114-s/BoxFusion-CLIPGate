# F3 FastSAM/OpenBox-lite projection shadow: frozen protocol

Status: preregistered before any F3 access to ScanNet ground truth.

## Purpose

F3 asks one narrow question after F2 was rejected: can causal multi-view
projection evidence turn the sealed F1/H0 residual candidates into better box
geometry?  It is an observer-only capacity experiment.  It does not add a
prediction, change a native prediction, or authorize birth.

The experiment is OpenBox-inspired, but it does **not** reuse the failed R2
replacement policy.  R2 operated on native terminal tracks and center-crop
depth fragments.  F3 operates on the exact sealed FastSAM masks and H0 points.

## Isolation and inputs

- Scene universe: the frozen paper100 prefix, in its sealed order.
- Candidate universe: exactly 52,299 F1/H0 sources from the completed F2
  sidecars and NPZ evidence.  `HL` and `HLG` are forbidden inputs.
- Allowed source fields: source/frame identity, FastSAM confidence (ordering
  and receipt only), raw mask packbits, H0 `q02/q98`, H0 world points and 5 cm
  voxel keys.
- Per-frame camera data: sealed ScanNet pose and depth intrinsics.  F3 projection
  shadow does not read RGB or depth pixels.  Pose, intrinsics, schedule, F0
  sidecar, F2 scene sidecar, and F2 NPZ are rehashed before use.
- Forbidden to the shadow runner: GT, evaluator, native terminal predictions,
  CLIP, labels/classes/semantics, training, online learning, and future-frame
  logical access.
- Loading a compressed scene NPZ may physically decompress the archive, but a
  guarded accessor exposes a source only when its sealed frame ordinal is not
  later than the current update.  The receipt records the maximum logical
  ordinal accessed.

## Causal association

Association reuses the already frozen target-masklift constants rather than
tuning on F3 GT:

- world AABB IoU at least `0.10`;
- center distance at most `0.50 m`;
- track TTL `10` scheduled keyframes;
- at most `1024` live tracks;
- at most `5` observations retained per track.

At the start of a frame, all matches are computed against a snapshot committed
at the end of the previous frame.  Edges are sorted by decreasing AABB IoU,
then increasing center distance, past track ID, and current source ID.  Greedy
one-to-one assignment follows that order.  Current sources cannot match one
another, a track accepts at most one source in a frame, track IDs are monotonic
and never reused, and there is no retrospective merge.  An unmatched source
may create an internal shadow track; that is not a prediction birth.

After five observations, the bounded memory keeps the five most recent
distinct-frame observations.  Empty and failed scheduled frames advance TTL.
A track is eligible for geometry only with at least three retained distinct
frames.

## Bounded H0 evidence

Each observation retains its H0 AABB, mask, pose, source identity, and at most
512 unique 5 cm voxel keys.  Unique keys are lexicographically ordered; if the
cap is exceeded, 512 inclusive linearly spaced positions are retained.  A
track therefore retains at most 2,560 voxel keys plus five packed masks.

## Dual hypotheses

### B: best single view

Every retained H0 AABB is projected into every *other* retained view.  Its
score is the median projected-AABB/mask IoU over valid other-view projections.
At least two valid held-out views are required.  The highest score wins; exact
ties choose the earlier frame and then lexical source ID.  A box is never
scored on the mask that generated it.

### C: multi-view consensus

For each leave-one-view-out fold, only the other retained views are used.
Their 5 cm voxel union retains a voxel when it has Chebyshev-radius-one support
in at least two distinct fitting views.  The lexicographically ordered retained
voxel centers are capped at 2,048 and a fixed world-axis `q02/q98` AABB is fit.
The fold box is evaluated only on the held-out raw mask.  At least 16 consensus
voxels, extent at least `0.02 m` on every axis, and two valid folds are required.
The C score is the median held-out projected-AABB/mask IoU.  The terminal C box
uses all retained views after the LOO score has been fixed.

Projection uses all eight float64 AABB corners, the sealed depth intrinsics and
world-to-camera inverse pose.  A hypothesis is invalid in a view unless all
corners are beyond the `1e-3 m` near plane.  Its clipped continuous XYXY box is
rasterized with floor starts and ceil stops against the exact 480x640 binary
mask.  F3 intentionally adds no depth/free-space gate: this keeps projection
self-validation as the only new geometry variable.

## Fixed no-GT selector

- B is valid with at least two folds and median projection IoU `>=0.10`.
- C additionally requires all LOO/full consensus AABB IoUs to have median
  `>=0.25`, center shift from B no more than `0.50 m`, per-axis extent ratio in
  `[0.5, 2.0]`, and volume ratio in `[0.25, 4.0]` when B is available.
- Choose C only when it is valid and its score is at least `B + 0.03`.
- Otherwise choose valid B.  If B is invalid, choose valid C.  If neither is
  valid, abstain.

Both hypotheses and the fixed selector are sealed in shadow receipts.  The
selector still cannot create an output box in F3.

## Runtime and causality gates

- Incremental F3 mean `<=25 ms/keyframe`, p95 `<=40 ms/keyframe`.
- Gap-25 amortized F3 cost `<=1.0 ms/source frame`.
- Composed frozen FastSAM/F0 plus F3 complete p95 `<=250 ms/keyframe`, maximum
  `<833.33 ms/keyframe`, and gap-25 amortized total `<=10 ms/source frame`.
- New GPU allocation is zero; total inherited GPU peak remains `<=4 GiB`.
- Prefix-invariance, query-before-commit, one-source/one-track, and maximum
  logical accessed ordinal checks must all pass.

Archive decode, input hashing, identity validation, JSON/NPZ serialization,
and GT oracle time are audit overhead and are reported separately from the
online incremental cost.

## GT-assisted oracle after sealing

Only after all paper100 shadow receipts are create-only sealed may a separate
oracle read GT and native predictions.  It reports B-only, C-only, fixed
selector, and identity-constrained grouped `(B or C)` capacity.  One shadow
track can contribute at most one hypothesis and match at most one GT; one GT
can be matched once.  Strict IoU comparisons are `>0.15`, `>0.25`, and `>0.50`.

The oracle must first reproduce the official constant-score native baseline
and the exact F1/H0 source identities.  Native rows, order, geometry and formal
score `1.0` remain unchanged.  Any GT-selected suffix is explicitly
non-deployable.

As a separate no-GT-selector diagnostic, the oracle also appends every
non-abstaining fixed-selector track in scene/track-ID order, each with formal
score `1.0`, and runs the same official class-agnostic evaluator.  This reports
the AP/TP/FP effect of the frozen selector without GT selection.  It remains a
shadow counterfactual—not an active F3 output—and is never substituted for the
identity-constrained capacity gate.

F3 is retained for the next selector/filter experiment only if its grouped
AP50 native-union capacity is at least 78 additional matches (F1/H0's 63 plus
the preregistered gain of 15), while reporting the final `144 matches / +10 AP
points at every threshold` goal separately.  F3 never authorizes active birth;
passing only authorizes a new preregistered active-gate experiment.

## Pre-GT implementation erratum (2026-08-29)

The first no-GT paper100 shadow pass sealed all 100 scene receipts but stopped
before either shard manifest was written.  The production census guard used
the upstream F2 names (`keyframes`, `successful_frames`, `sources`) to index
F3's normalized scene-count schema (`keyframe_count`,
`successful_frame_count`, `source_count`), so each missing `Counter` entry was
read as zero.  The guard keys were corrected to the existing F3 schema and a
regression test was added.  No threshold, hypothesis, selector, input,
runtime definition, scene order, or retention rule changed; GT, predictions,
the evaluator, and the F3 oracle had not been accessed.  Receipts from the
stopped pass are archived, and the production shadow is rerun create-only
under the corrected runner/protocol hashes before any oracle access.
