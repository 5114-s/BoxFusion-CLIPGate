# S2 frozen-YOLOE direct-proposal shadow preregistration

Date: 2026-08-23

## Decision being tested

The earlier Graw, Gclean/SMOV, PUF-lite and Boxer-Past3 branches had too few
correct new proposals to support a +10 absolute-AP target.  S2 therefore tests
the first structurally different branch: a complete frozen prompt-free
instance-segmentation proposal source, lifted directly from the current RGB-D
stream and accumulated in bounded past/current-only geometry memory.

S2 is shadow/counterfactual only.  It is not authorized to alter the completed
T05 files or create an active birth.  Candidate thresholds in this document
were fixed before reading S2 ground truth.  One no-GT engineering smoke run on
`scene0377_02` was used only to establish that the provider loads and that a
sidecar can be written; its output is not a sealed S2 result.  The formal dev3
run is regenerated create-only under a new frozen artifact root.

## Fixed native protocol and scene order

- Native prefix: `results/scannet_topk_fusion_score05`, produced with
  `score_thresh=0.5`, appearance gate disabled and Reliable-View Top-K3.
- Formal evaluation score: constant `1.0` for every native and appended row.
- Frozen CuTR and original frozen CLIP vocabulary/features remain unchanged.
- S2 detector labels are diagnostic only and cannot be used for selection,
  ranking, class assignment, or evaluation.
- Development scene order is exactly `scene0568_00`, `scene0606_01`,
  `scene0377_02`, from
  `evaluation/data_util/meta_data/scannetv2_graw_e2_preflight3.txt`.
- Scene-list SHA-256:
  `117b5bea04c557f52d4c2a9435c3961bbaae66e420fb5bb849a278f89fe454fc`.

The exact no-GT RGB/depth/pose/calibration schedule was sealed before the
formal run.  It contains 209 scheduled keyframes across three scenes:

- seal: `logs/scannet_s2_yoloe_direct_shadow_score05/dev3_stream_input_seal.json`;
- seal SHA-256:
  `cf363f9d92bd5b0c1aaa51ee6c200744fbf60404d671320c75df96ae20128655`;
- the seal states `gt_access=false` and is verified again before inference.

## Fixed causal S2 branch

For each scheduled keyframe, in stream order:

1. Frozen YOLOE-11s prompt-free segmentation sees only the current RGB image:
   confidence `0.25`, NMS IoU `0.70`, image size `640`, mask threshold `0.50`,
   class-agnostic NMS and at most 64 masks.
2. Each mask is intersected with current valid metric depth in `[0.10, 6.00]`
   m.  A one-pixel mask-edge margin and `0.15 m` four-neighbour depth-edge
   removal are applied.
3. Points are transformed by the current camera pose, voxelized at `0.02 m`,
   bounded to 2,048 points per observation and 8,192 points per track, and
   fitted with the fixed `q02/q98` world AABB.
4. Geometry-only tracking accepts either AABB IoU at least `0.05`, or center
   distance at most `0.75 m` together with bidirectional maximum containment at
   least `0.25`.  One track can receive at most one observation per frame.
   Candidate TTL is 10 successful provider calls; confirmed tracks are stored
   in a bounded archive.
5. A candidate needs at least three distinct past/current keyframes, mean
   frozen-provider score at least `0.25`, mean projection consistency at least
   `0.30`, and all three AABB extents at least `0.30 m`.

No future frame, terminal backfill, target-dataset fit, gradient update,
learned quality head, learned box refiner, CLIP feature, ScanNet label, GT box,
or oracle report is available to these steps.  Native CuTR inference inside
the nested runner exists only to drive the observer lifecycle.  Its slightly
nondeterministic diagnostic final boxes are discarded and are never treated
as the formal T05 prefix.

## Fixed terminal shadow materialization

The no-GT materializer reads the completed T05 pickle separately and performs
only these deterministic operations on the already emitted S2 candidates:

- reject a candidate if its maximum AABB IoU with any frozen T05 box is at
  least `0.10`;
- candidate-to-candidate NMS at AABB IoU `0.25` in the emitted deterministic
  ranking;
- retain at most 6 candidates per scene;
- preserve every native T05 row, class, geometry, score and order byte-for-byte
  as the prefix; any candidate is a suffix only;
- record `birth=false`, `active_authorized=false`, `gt_access=false`, input
  hashes before/after, and the constant-`1.0` formal score contract.

No threshold may be changed after looking at the dev3 S2 result.  Alternative
quantiles, extent thresholds, detector confidence values, caps, or semantic
gates require a new named branch and a new untouched split.

## Frozen implementation ledger

| Component | SHA-256 |
|---|---|
| `config/scannet_s2_yoloe_direct_shadow_score05.yaml` | `4f3e9739b296197d41c0d322c0a1e30230385ccb8c1384a36615ffa413e83441` |
| YOLOE-PF checkpoint | `292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d` |
| nested `demo.py` | `57fb58596401324785ee9696d16ebc15eed082df00dd6afede9e6d440b217423` |
| `online_refinement.py` | `0faf3d7d6242facdd9300a942fe1e2bf2364f5f9ebc17e8f8f278382a0102f61` |
| `object_memory.py` | `c2f3f0e0753a34430f0d9d03c65039aa6eee80114a1337676ec4b5f1eaa60938` |
| `supplemental_proposals.py` | `dcab601eb7bd70328be882e8944619e4dffd6d366214dd74eb6c2d5a3cfc001d` |
| `tr3d_c2_maskrgbd_observer.py` | `108e4c1684a6f5e3b352b31a9d6e026e393bc1872540653312e7bdfb0d1e4778` |

The checkpoint is frozen and is not trained or fine-tuned on the target
ScanNet split in this experiment.  S2 contains no train command and no mutable
model state.  The materializer must fail closed if any ledger digest differs.

Frozen native dev3 prediction SHA-256 values are:

- `scene0568_00`: `b55ce48fb6eb4dad9ee5bfe7007c3dbc9898b3f72ddbc5ad428b8be6414bcd2d`;
- `scene0606_01`: `d4e8d6dc85c917ac1634b81a45adb3866279d3e02f470c43b23bd71f5bb3ef1c`;
- `scene0377_02`: `ed7f849a33d45eebe846559a90aeb7de1a97f2eb169c3a7c0cb5de61d3dab35b`.

## Promotion and stopping rules

1. Run and seal the fixed S2 sidecar on dev3 without GT.
2. Open dev3 GT only after the sidecar and all hashes are fixed.  Promotion to
   H10 requires, at every IoU `0.15/0.25/0.50`: (a) strictly positive
   constant-score AP delta and (b) at least one additional maximum-cardinality
   match over native T05.  Any threshold failure rejects this S2 branch.
3. H10 remains one-shot and unopened for S2 unless dev3 passes.  Its fixed list
   SHA-256 is
   `8965d0534ed3028f85d8b0ea7227d348a6faa1387b858ddf42c3183bd9ebdf90`.
   H10 promotion requires nonnegative AP delta and at least one additional
   maximum-cardinality match at all three IoUs, with no hash, cap, causality,
   or runtime-contract failure.
4. If H10 passes, freeze the branch unchanged and run C87, then report full100
   only as a secondary aggregate.  C87 list SHA-256 is
   `3fb0f8bc79217cfe3ce47bf05970b3a4f75981e357a50a7804cf51f0e4c77b2c`.

Passing dev3/H10 establishes usefulness, not a +10 result.  A +10 absolute-AP
claim requires the unchanged full100 constant-score evaluation.  Online
feasibility is also mandatory: bounded memory, no future-frame access, and an
end-to-end same-hardware throughput check against frozen T05 before activation.
