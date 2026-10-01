# F6 GT-free past-only multi-view selector — protocol freeze

Frozen: 2026-08-29 (Asia/Shanghai), before any F6 execution or F6 access to
ScanNet ground truth/evaluator output.

Protocol ID:
`F6-GT-FREE-PAST-ONLY-MULTIVIEW-DEPTH-PROJECTION-SELECTOR-PAPER100`.

## Question and experimental status

F5 retained too little of F4's frozen four-hypothesis geometry capacity at
IoU 0.50.  F6 is one new, locked exploratory attempt to select a geometry
using actual cross-view mask and depth evidence instead of Boxer confidence
or a single-frame rule.  F6 was designed after the aggregate F5 result was
known, so paper100 is not an untouched confirmation set; any successful rule
must later be repeated unchanged on held-out scenes.

F6 directly consumes the sealed F2/F4 evidence and does not consume F5
choices.  It remains shadow-only.  For every one of the same 52,299 sources it
copies exactly one existing `H0/HL/HLG/HB` geometry, keeps formal score `1.0`,
and cannot add/drop a source, create a birth, read or mutate native BoxFusion
predictions, or change CLIP, class, embedding, rank or source order.

Frozen schemas:

- `boxfusion.fastsam_f6_mvdc_selector.v1`;
- `boxfusion.scannet_fastsam_f6_mvdc_paper100.scene.v1`;
- `boxfusion.scannet_fastsam_f6_mvdc_paper100.shard.v1`;
- `boxfusion.scannet_fastsam_f6_mvdc_paper100.merge.v1`.

## Sealed input boundary

- paper100 scene-list SHA-256:
  `4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5`;
- F4 no-GT merge SHA-256:
  `0e00ab68e2525b8e1262dfb12bc08ee3a98f02d70b158960f49379e957f826a6`;
- upstream F2 no-GT merge SHA-256:
  `455c0e36e35a30c7ba5915384e4d159a730a47b3368bf4b3fb6a5f6064f25603`;
- exact source census: 100 scenes, 6,817 scheduled keyframes, 6,726
  successful frames and 52,299 sources in sealed scene/frame/rank order.

Allowed inputs are source identity/lineage, F2 packed FastSAM mask and original
world points, `H0/HL/HLG`, F4 `HB`, tight mask box, current pose and sealed
depth-camera intrinsic.  All four hypotheses use the same original F2 point
evidence.  Boxer confidence/log-variance may be copied as diagnostics but may
not influence selection.

Forbidden inputs are GT/annotations, evaluator code or output, every F1--F5
oracle/match ledger, native/terminal predictions, scene axis alignment,
labels/semantics/CLIP, future frames, ScanNet-specific fitting, training,
fine-tuning, calibration, optimizer state, online learning and a directory or
nearest-frame search.  An attempted forbidden read is fatal.

## Deterministic bounded evidence

F2 masks are authenticated 480x640 little-endian packed bit arrays.  Positive
mask pixels are enumerated row-major.  A point or positive-pixel list with
`N<=256` is retained in full.  Otherwise retain indices

`floor((j+0.5)*N/256), j=0,...,255`.

No random sampling or source-dependent seed is allowed.  Each committed past
source stores only its H0 AABB, at most 256 original world points, at most 256
positive mask-pixel coordinates, the packed mask, pose and identity.  State is
limited to the previous three successful scheduled frames, at most 16 sources
per frame and at most 2.5 MiB of raw array payload.  F6 allocates no CUDA tensor.

## Causal H0-only association

Association is recomputed by F6 and never imports F3 track IDs or F5 selected
geometries.  A current H0 and a source H0 in one committed past frame form an
eligible edge when normalized center distance `ND<=0.50` and either world-AABB
`IoU3D>=0.15` or symmetric containment `SC>=0.60`.  Affinity is the exact
lexicographic tuple `(IoU3D, SC, -ND)`.  Keep only mutual-best edges within
each past frame; exact ties prefer lower rank and then lexical source ID.

For each current source retain its matches from the two most recent distinct
past frames.  Fewer than two matches means strict fallback to `Hbase`.  All
current sources query one immutable past snapshot; the current frame becomes
visible only after an exact-token commit.  Thus current sources cannot support
one another and maximum lookahead is zero.

## Geometry and conservative fallback

`Hbase` exactly reuses the frozen F5 base rule without using F5 output:
`HLG`, then `HL`, is eligible only when applied without fallback, retains at
least `max(16,ceil(0.55*n0))` points, has volume ratio to H0 in `[0.25,1.05]`,
per-axis extent ratios in `[0.35,1.05]`, and center shift no greater than
`0.20*diagonal(H0)+0.05 m`; otherwise use H0.

World AABBs use identity local rotation.  HB is evaluated in its sealed OBB
frame `u=R^T(p-c)` and is valid only when F4's finite, positive-extent,
right-handed/orthonormal, eight-corner and positive-camera-depth checks
reproduce.  For coarse comparison only, HB uses the min/max envelope of its
sealed eight world corners.  HB confidence is not a gate.

## Three fixed multi-view metrics

The three views are the current source and its two causal past matches.  A
hypothesis originating at the current source is scored against the mask and
original points of all three views.

1. **Depth containment.**  In the hypothesis local frame, record per-view
   exact containment `C0_v` and containment after expanding every local face
   by exactly 0.05 m, `C5_v`.  Aggregate `C0` and `C5` by the median.
2. **Robust face residual.**  Concatenate the equally bounded point samples
   from the three views.  In the hypothesis local frame compute q02/q98 using
   NumPy linear quantiles.  `D` is the mean absolute displacement between
   these six quantile faces and the six hypothesis faces, in metres.
3. **Projection-mask consistency.**  Project the true eight corners through
   each view; all corners must be beyond `1e-4 m`.  Let their 2D convex hull be
   the projected polygon.  `R` is the fraction of at most 256 sampled positive
   mask pixels inside the hull.  For `P`, place a fixed 16x16 cell-centre grid
   over the clipped hull bounding rectangle, keep centres inside the hull and
   query their exact packed-mask bits.  Define `J=0` if `P*R=0`, otherwise
   `J=1/(1/P+1/R-1)`.  Aggregate `J` by the median over the three views.

All computations are float64 and threshold comparisons are inclusive.

## Candidate gate and exact selector

A non-base candidate must be a valid copied F4 hypothesis, project validly in
all three views, and have at least 16 finite original points in every view.
Relative to Hbase its envelope must satisfy `ND<=0.50`, volume ratio
`[0.25,4.00]`, and either `IoU3D>=0.20` or `SC>=0.70`.  Its current-view
support must satisfy `C0_v>=0.60` and `C5_v>=0.80`, and at least two of three
views must satisfy both support thresholds.

Compare each gated candidate with Hbase:

- depth win: `D_candidate <= D_base - 0.05 m`;
- projection win: `J_candidate >= J_base + 0.10`;
- containment win: `C0_candidate >= C0_base + 0.10`.

It must win at least two metrics and must not regress any metric beyond:

- `D_candidate <= D_base + 0.025 m`;
- `J_candidate >= J_base - 0.05`;
- `C0_candidate >= C0_base - 0.05`.

Among passing candidates choose lexicographically by descending win count,
ascending D, descending J, descending C0, then `H0 > HL > HLG > HB`, which is
only an exact tie-break.  Otherwise copy Hbase exactly.  The 5 cm quantities
reuse the frozen F2/F5 robustness scale; 2.5 cm is half that fixed scale, and
the 10-point sampled-proportion margin is greater than three worst-case
standard errors for 256 probes.  These constants are not fit to paper100.

Every switch seals the two past source/frame identities and every raw
per-view scalar, candidate gate, win/non-regression test, selected geometry
hash and source lineage.  A failed optional hypothesis falls back; invalid H0,
mask, point, pose, intrinsic or lineage evidence is fatal.

## Integrity, causality, runtime and no-GT stopping rule

Inputs are hashed before and after.  A half-prefix replay and a second CPU
replay must reproduce every corresponding result hash.  Synthetic future
perturbation tests must leave earlier results unchanged.  Native/source/score/
class/semantic mutation, birth, future, GT, prediction, training and learning
counts must all be zero.

Frozen runtime gates (serialization, whole-scene archive inflation, hashing
and audit replays excluded) are:

- F6 incremental warm p95 `<=25 ms/keyframe`;
- replay-composed F0+F2+F4+F6 warm p95 `<=375 ms/keyframe`;
- composed warm maximum `<833.33 ms/keyframe`;
- composed warm mean/25 `<=15 ms/raw frame`;
- zero warm gap-25 deadline misses and inherited CUDA peak `<=4 GiB`;
- bounded-state limits above all pass.

A later active experiment still requires a same-GPU live end-to-end
measurement of at least 15 FPS; replay timing alone is insufficient.

F6 is retained for one separately sealed evaluation only when all integrity,
causality, determinism and runtime gates pass and non-base switches number at
least 144 across at least 20 scenes but no more than 20% of sources.  Outcomes
are fixed as:

- contract/runtime failure: `discard_f6_selector`;
- too few switches/scenes: `stop_f6_insufficient_multiview_switches`;
- more than 20% switches: `stop_f6_overbroad_switches`;
- otherwise: `retain_f6_for_one_separately_sealed_evaluation_only`.

Passing never authorizes birth, output mutation or deployment, and no rule may
be changed after paper100 F6 receipts are observed.
