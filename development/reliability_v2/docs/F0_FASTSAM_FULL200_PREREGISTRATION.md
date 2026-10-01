# F0 frozen FastSAM-x residual automatic-mask shadow — full200

## Objective and boundary

F0 tests whether a frozen, class-agnostic automatic-mask source has enough
causal proposal capacity to justify a later past-only selector.  It is a
shadow experiment: it does not track, create births, write prediction pickles,
change native boxes, change CLIP, or compute AP.  Passing F0 authorizes only a
separately preregistered oracle/selector experiment; it never authorizes adding
all masks to the constant-score output.

No training, target-dataset tuning, online learning, labels, annotations, GT,
terminal BoxFusion/Cbest boxes, future frames, or evaluator output may be read
by the F0 runner.

## Frozen 200-scene cohort

The exact list is
`evaluation/data_util/meta_data/scannetv2_val_f0_full200.txt`, SHA-256
`0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1`.
It contains the paper full100 in its released order followed by the first 100
entries of the canonical 312-scene ScanNet validation list that are not in the
paper full100.  This yields 200 unique official validation scene IDs.

The released first-100 CuTR-v2 cache remains read-only.  The extra 100 use an
independent cache root and the same frozen CuTR checkpoint, score threshold
0.5, UV/floor filtering, frame-zero retry, gap 25, and released early-exit
schedule.  PyTorch safely deserializes each complete cache payload, but F0
indexes and uses only the current frame's sealed `pred_boxes[N,4]`; it never
uses cached classes, logits, scores, descriptors, or 3D boxes.  Across full200
the frozen schedule contains 12,941 keyframes: 6,817 existing and 6,124 new.

## Frozen FastSAM provider

- Checkpoint:
  `/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/checkpoints/FastSAM.pt`
- Bytes: `144943063`
- SHA-256:
  `c0be4e7ddbe4c15333d15a859c676d053c486d0a746a3be6a7a9790d52a9b6d7`
- Environment: `boxfusion-online`, Ultralytics `8.4.105`, PyTorch
  `2.6.0+cu124`; parameters are frozen and inference-only.
- Input: one current 640x480 uint8 BGR frame.
- Call: `imgsz=1024`, `conf=0.25`, `iou=0.90`, `max_det=100`,
  `agnostic_nms=True`, `retina_masks=True`, `classes=None`, `augment=False`,
  `half=False`, `batch=1`, `verbose=False`, `save=False`, `stream=False`.
- Masks are thresholded at 0.5 and must return to 480x640.  Provider class IDs
  and names are ignored; confidence is used only for deterministic shadow
  ordering and never becomes a detection score.

## Current-only residual membership

Each current CuTR 2D box is expanded by four pixels and clipped to 640x480.
Their order-invariant union is `E`.  For a FastSAM mask `M`, valid metric depth
is `V = finite(depth) & 0.10 <= depth <= 6.00`; `S=M&V` and `R=S&~E`.

A mask is eligible only when all of the following hold:

- 200 <= mask pixels <= 122,880;
- tight-box side >=16 pixels and aspect <=6;
- valid-depth ratio >=0.50;
- residual pixels >=200 and residual ratio >=0.20.

Residual support is only a membership test.  Three-dimensional lifting uses
the complete valid mask, never the residual fragment.  This distinction avoids
the geometry damage observed with earlier hard unexplained-depth gates.

Eligible masks are sorted by `(-confidence, -residual_ratio,
-residual_pixels, tight_box, mask_sha256)`.  Greedy duplicates are suppressed
at mask IoU >=0.80 or smaller-mask containment >=0.90.  At most 16 masks per
frame are lifted; all drops are counted.

## Frozen geometry

Full-mask support removes a one-pixel mask boundary and both endpoints of a
four-neighbour depth jump greater than 0.15 m.  Current depth, intrinsics, and
the current rigid-valid pose backproject support to world coordinates.  An
invalid current pose always abstains; F0 never substitutes a historical pose.
The frozen full200 census contains exactly 229 such keyframes.  The CuTR
producer-orientation census contains 12,939 UPRIGHT frames, one LEFT frame
(`scene0246_00/1900`), and one RIGHT frame (`scene0426_00/2200`).  Because the
last two use a 640x480 cache coordinate frame while the provider/core contract
is 480x640, they also abstain rather than applying an unregistered inverse
transform.  Thus exactly 12,710 keyframes are eligible to call the provider.
Geometry uses 2 cm voxels, requires 16 unique voxels, stores at most 2,048
deterministically selected points, and reports a q02/q98 world AABB with a
0.02 m minimum dimension.  F0 keeps no cross-frame object state.

## Shadow receipts and gates

Every provider row has a disposition.  Per-frame, per-scene, and total
receipts report provider calls/failures, raw masks, all filter reasons,
deduplication, Top-16 drops, accepted lifts, 100-mask saturation, accepted
scene coverage, runtime, memory, hashes, and bounded 3D geometry.  Sidecars are
create-only and output-inert.

The full200 no-GT capacity gate requires all of:

- at least 1,500 accepted lifts;
- accepted lifts in at least 160 scenes;
- Top-16 saturation in at most 25% of successful keyframes;
- zero input, causality, ledger, or output-mutation violations.

The production runner requires CUDA, PyTorch `2.6.0+cu124`, OpenCV `4.6.0`,
Ultralytics `8.4.105`, an RTX 3090, and compute capability 8.6.  Each shard
records and independently validates its GPU UUID.  The first three successful
provider calls of a fresh shard are excluded as warm-up.  A resumed process
performs three explicit, output-inert re-warm calls before its first pending
scene; those calls are kept outside both the 12,941-frame ledger and all
capacity/runtime distributions.

The isolated runtime gate on RTX 3090 requires provider p95 <=200 ms, complete
F0 keyframe p95 <=250 ms, complete maximum <833.33 ms, mean overhead divided by
25 <=10 ms per raw stream frame, and peak allocated memory <=4 GiB.  These are
shadow-branch gates, not proof of same-GPU integrated BoxFusion throughput.

Only after sealing may a separate read-only paper100 oracle open native boxes
and GT.  To retain a necessary ceiling for a +10 AP-point target over 1,433 GT
objects, union maximum matching must add at least 144 matches independently at
IoU 0.15, 0.25, and 0.50.  Even that result would authorize only a past-only
F1 selector, because it does not price false positives under constant scores.

## Model provenance caveat

The checkpoint is a generic FastSAM-x/YOLOv8x-seg model trained outside
ScanNet and is not fine-tuned on this target dataset, satisfying the requested
frozen-general-model condition.  For deployment, licensing needs separate
review: the installed Ultralytics package is AGPL-3.0, while FastSAM materials
describe the model separately and SA-1B has its own usage terms.
