# UDC + frozen MobileSAM v1 full100 preregistration

Date frozen: 2026-08-25 (before inspecting any UDC-v1/GT overlap or UDC-v1
active AP result)

## Purpose

This experiment tests whether a class-agnostic unexplained-depth component
(UDC) source can supply genuinely new proposals that are absent from the
frozen Cbest native prefix.  It is an inference-only branch:

- no ScanNet training, fitting, calibration, or online parameter update;
- frozen CuTR, Boxer, MobileSAM, and native CLIP weights;
- causal RGB-D, pose, and proposal observations only in the shadow producer;
- native BoxFusion association, geometry, order, scores, categories, and CLIP
  embeddings remain unchanged;
- an active result may only append surviving UDC boxes after the complete
  native prefix, with the official constant-score evaluator setting every
  score to exactly `1.0`.

MobileSAM is a generally pretrained frozen model.  Consequently this branch
is **target-dataset-training-free**, not a claim that MobileSAM itself was
never pretrained.

## Evidence boundary of the executable v1 replay

The current sealed replay contains a per-keyframe post-filter CuTR proposal
record, but it does not contain an immutable snapshot of BoxFusion's complete
live native memory at every keyframe.  V1 therefore uses only:

```text
current RGB-D + current pose
+ current keyframe's sealed post-filter CuTR 2D boxes
+ UDC state committed at earlier keyframes
```

to produce unexplained-depth components.  In particular, the shadow producer
must not open or accept an argument for:

- `results/scannet_t05_boxer_replay_active_score05/*_boxes.pkl`;
- any other terminal native prediction;
- ScanNet annotations, GT boxes, label meshes, or evaluator code.

For a keyframe `t`, a sampled depth pixel is explained iff its pixel centre is
inside the union of the current cached `pred_boxes` rows after expanding each
side by exactly 4 pixels:

```text
x1 - 4 <= u <= x2 + 4 and y1 - 4 <= v <= y2 + 4
```

The cache rows are already the frozen post-filter rows, including the released
frame-zero retry behavior, so v1 must not apply a second score threshold.  An
empty cache record explains zero pixels.  The fixed four-pixel margin absorbs
registration and stride quantization at the box boundary; it is not fitted on
evaluation results.

This coverage is strictly causal, but it deliberately omits objects committed
by native BoxFusion on earlier frames.  It therefore under-represents
*historical* native explanation.  Conversely, a rectangular 2D box can remove
background pixels inside that rectangle.  Neither effect may be described as
the exact live Cbest residual.

A later integrated version may replace this approximation by a copied
`all_pred_box(t)` snapshot taken inside the online loop.  That would be a new
protocol and must not be silently substituted into v1.

## Frozen inputs and schedule

- Scene order: the 100 entries in the official100 scene ledger.
- Cadence: the exact sealed `scannet-score05-gap25-postfilter-v2` manifests;
  no skipped keyframe may be replaced by a later frame.
- Sealed metadata totals: 100 scenes, 6,817 keyframes, 23,651 post-filter CuTR
  proposal rows, and 575 empty keyframes.
- RGB-D: ScanNet 640 x 480 color/depth frames; metric depth is obtained by
  division by `1000.0`.
- Pose and depth intrinsics: the corresponding frame pose and
  `intrinsic_depth.txt`.  A non-finite pose is replaced only by the most recent
  earlier valid pose, matching the causal ScanNet replay loader; if no earlier
  valid pose exists the keyframe explicitly abstains.  A missing or invalid
  intrinsic explicitly aborts that scene.
- MobileSAM checkpoint:
  `/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/pcdet/models/backbones_3d/focal_sparse_conv/MobileSAM/weights/mobile_sam.pt`,
  SHA-256
  `6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f`.
- MobileSAM uses box prompts only, `multimask_output=True`, and chooses the
  hypothesis with maximum predicted IoU.  No point prompt, mask prompt, GT
  prompt, automatic-mask grid, or test-time adaptation is permitted.

A read-only, no-GT raster audit of the sealed CuTR boxes found mean current-box
union coverage `0.4356` at stride four.  Of 6,817 keyframes, 2,029/3,927/5,747
have union coverage below `0.25/0.50/0.75`.  This is only a loose input-capacity
proxy; it is not object recall, precision, or AP evidence.

## Frozen depth preprocessing

At each valid keyframe:

1. Sample depth on a four-pixel grid.
2. Retain finite metric depth in `[0.10, 6.00]` metres.
3. Mark both endpoints of every valid full-resolution horizontal or vertical
   neighbour pair whose depth differs by more than `0.15 m`, dilate that mask
   with a fixed `7 x 7` square before stride-four sampling, and additionally
   reject both endpoints of any remaining stride-grid neighbour jump above the
   same threshold.  The full-resolution dilation prevents a discontinuity at
   pixel phase 1--3 from disappearing when the grid is sampled at phase zero.
4. Remove samples explained by the current post-filter CuTR 2D union defined
   above.
5. Back-project the remaining samples and transform them to ScanNet world
   coordinates using only the current pose.
6. Quantize accepted-component points with signed floor at `0.05 m` for mask
   proximity and cross-view support bookkeeping.

No terminal native box is used in these steps.

## Frozen executable component and structural rejection

Executable v1 deliberately uses a small deterministic image-grid component
extractor instead of a fitted plane model.  The remaining stride-four binary
grid is split with 8-connectivity.  This makes every prompt retain an exact
set of source depth pixels, while the 0.15 m edge rejection prevents direct
connections across strong depth discontinuities.

A component is eligible only when all conditions hold:

- 24 to 5,000 stride-grid pixels;
- grid bounding-box width and height are each at least 5 samples;
- it touches at most one image-grid border;
- its world q02/q98 AABB extent on every axis is in `[0.05, 2.50] m`;
- its world q02/q98 AABB diagonal is at most `3.00 m`;
- its world q02/q98 AABB volume is at most `4.00 m^3`.

These extent and border gates are the executable v1 structural rejection for
large wall/floor regions.  No RANSAC, learned plane head, normal-bin fitting,
or GT-selected structural parameter is used.  Components are ranked by more
grid pixels, then top-to-bottom, left-to-right, grid height/width and component
ID; only the first two are offered to MobileSAM.  Each selected component
retains both its source pixel indices and its signed-floor 5 cm voxel keys.

## Frozen bounded component tracker and prompt budget

Raw components are associated one-to-one to past raw tracks when centre
distance is at most `0.50 m` and either AABB IoU is at least `0.05` or maximum
directional containment is at least `0.30`.  Valid pairs are ordered by higher
IoU, higher containment, shorter centre distance, lower track ID, and lower
lexicographic component key.  No semantic feature or learned association is
used.

The memory is bounded to:

- TTL 250 source frames, equivalent to ten gap25 keyframes;
- at most 128 live raw tracks per scene;
- at most 4,096 lexicographically retained 5 cm voxels per track.

Expired entries are removed first.  Capacity eviction removes the oldest
`last_seen`, then the lower track ID.  This rule must be reported if it fires.

At most two components prompt MobileSAM at one keyframe.  Selection is made
before tracking and therefore cannot depend on a future confirmation state:
the two largest eligible current components are used, with the fixed
top-to-bottom and left-to-right tie break above.  The raw tracker consumes
those selected components and never feeds a learned score back into prompt
generation.

## Frozen MobileSAM prompt, mask, and lifting gates

The prompt is the tight current-frame component pixel box, expanded on every
side by `max(8 px, 0.10 * corresponding_side)`, then clamped to the 640 x 480
image.  It must have:

- minimum side 16 pixels;
- area from 400 pixels through `0.40 * H * W`;
- aspect ratio at most 6.

Pixels outside the expanded prompt rectangle are set to false before lifting
and component-coverage measurement.  The selected MobileSAM mask must satisfy:

- predicted IoU at least `0.80`;
- mask area from 200 pixels through `0.40 * H * W`;
- coverage of the raw component's source pixels at least `0.50`.

Mask depth is filtered by the same valid-depth and depth-edge rules.  A lifted
mask point is retained only when its 5 cm voxel is within Chebyshev distance
two (10 cm) of a current raw-component voxel.  The retained points use the
existing frozen object-memory geometry rules:

- 2 cm observation downsampling;
- at most 2,048 points per observation;
- world AABB lower/upper quantiles `0.02/0.98`;
- at least 16 supported 5 cm voxels per accepted view.

## Frozen past-only confirmation

Lifted observations inherit their causal raw-track identity.  A track becomes
a shadow birth candidate only after all of the following are true using
observations no later than the current keyframe:

- three distinct, strictly increasing keyframes;
- first-to-third frame span at least 50 source frames;
- camera-centre baseline at least `0.15 m`;
- maximum viewing-ray separation at least 8 degrees;
- minimum/median MobileSAM predicted IoU at least `0.80/0.85`;
- median pairwise lifted-AABB IoU at least `0.15`;
- maximum pairwise lifted-AABB centre distance at most `0.40 m`;
- fused support at least 32 unique 5 cm voxels;
- fused `q02/q98` AABB extents each at least `0.05 m`, maximum extent at most
  `2.50 m`, diagonal at most `3.00 m`, and volume in
  `[0.001, 4.00] m^3`.

The birth time is the third confirming keyframe.  Later evidence may update a
shadow track but may not retroactively move its birth earlier.

## Terminal novelty and the online-claim boundary

The executable v1 shadow cannot perform exact live-native novelty because the
replay lacks per-keyframe native-memory snapshots.  After the entire no-GT
shadow and all thresholds above are frozen, the active materializer may read
the final Cbest prediction only for these terminal operations:

1. preserve it as an unchanged, byte-equivalent native prefix;
2. reject a UDC candidate only when candidate-in-native or
   native-in-candidate AABB containment is at least `0.80`; native IoU is
   recorded as a diagnostic but is not a hard gate;
3. apply deterministic UDC self-NMS at AABB IoU `0.15` or either directional
   containment `0.50`;
4. retain at most four appended UDC boxes per scene, ordered by higher mean
   predicted IoU, more fused voxels, earlier birth time, then lower track ID.

The final-prefix check must not feed back into historical component creation,
prompt selection, mask lifting, tracking, or confirmation.  It is a
future-aware terminal deduplication step.  V1 must therefore report:

```text
causal_shadow_generation = true
strict_online_native_novelty = false
terminal_replay_materialization = true
```

The v1 AP experiment is a causal proposal-generation replay plus terminal
materialization.  It must not be presented as proof of an integrated
per-keyframe live Cbest+UDC system.

Native CLIP categories, vocabulary, and embeddings are not changed.  The UDC
suffix is class agnostic for this ScanNet evaluation; no result may be used to
claim a semantic improvement.

## Capacity diagnostics and mandatory full100 active evaluation

Before GT or evaluator access, report:

- total raw components and scene coverage;
- prompt count, accepted MobileSAM masks, and every rejection reason;
- confirmed pre-novelty tracks;
- terminal-novel and post-self-NMS candidates;
- candidate scene coverage and per-scene cap count;
- memory eviction and invalid-input abstention counts.

The preregistered capacity reference is:

- at least 300 confirmed pre-novelty tracks;
- at least 250 terminal-novel/post-NMS candidates;
- candidates in at least 80 scenes.

Passing this reference only means that raw proposal count is not an immediate
mathematical bottleneck.  It does not estimate precision or AP.  Failing it is
reported as a failed capacity diagnostic, but **does not stop** the requested
active full100 materialization and AP evaluation.  The full100 active result
must still be produced so that a low-capacity but high-precision branch is not
discarded without measurement.

The official100 evaluation is run only after the no-GT shadow artifact and
terminal policy are sealed.  It uses:

- baseline `results/scannet_t05_boxer_replay_active_score05`, 1,788 native
  boxes;
- `score_thresh=0.5` in native proposal production;
- final confidence exactly `1.0` for every native and appended prediction;
- official AP15, AP25, and AP50 on the same 100 scenes.

Report absolute AP changes, prediction count, recall and precision changes,
and the number and marginal precision of appended matches at every threshold.
No threshold or category selection may be changed after viewing these values
and then reported as the same v1 experiment.

## Runtime diagnostics

Runtime is reported separately from AP and does not block the requested
full100 AP run.  Measure after warm-up, batch size one:

- UDC preprocessing (`coverage + normals + planes + components`) mean/p50/p95;
- MobileSAM provider and mask-lifting mean/p50/p95;
- complete incremental keyframe mean/p50/p95;
- prompts per keyframe and peak CPU/GPU memory.

Reference engineering limits are UDC preprocessing p95 at most 60 ms,
complete incremental mean/p95 at most 150/250 ms per keyframe, and mean
amortized overhead at most 6 ms per original stream frame under gap25.  A later
live fixed-scene integration should target median at least 20 FPS and worst
scene at least 15 FPS.  Replay timing alone is not end-to-end live FPS proof.

## Required artifact contract

The shadow JSON/NPZ must record, per scene and keyframe:

- scene/frame ID and hashes of RGB, depth, pose, intrinsics, proposal cache,
  MobileSAM checkpoint, and relevant source files;
- current CuTR 2D-box count, valid/explained/residual depth counts;
- structural-bin and component counts;
- deterministic component/track IDs and state-before/state-after summaries;
- prompt, MobileSAM, lifting, confirmation, and rejection metrics;
- all frozen policy constants and runtime samples.

The shadow process exposes no annotation, GT, terminal-prediction, or evaluator
argument.  The materializer exposes no annotation or evaluator argument and
may read the terminal Cbest root only for immutable-prefix copying and the
terminal novelty rule above.  Native input hashes are recorded before and
after both stages and must remain identical.
