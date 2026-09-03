# Target-first MobileSAM mask-lift: official100 preregistration

Frozen on 2026-08-25 (Asia/Shanghai), before evaluating this branch on
ScanNet official100.  This is an exploratory, no-target-training experiment;
it does not erase knowledge of earlier, different Raw-Boxer experiments.

## Immutable comparison protocol

- Native prefix: `results/scannet_t05_boxer_replay_active_score05`.
- Input detector threshold: `score_thresh=0.5` in the sealed Cbest replay.
- Evaluation: the fixed official100 scene list and the constant-score
  evaluator, which assigns every final prediction score `1.0`.
- Every native row (class, corner bytes, score, order) must be preserved.
- The frozen CuTR/Cubify Anything, BoxerNet, OWLv2, MobileSAM and native CLIP
  checkpoints are inference-only.  There is no ScanNet training, fine-tuning,
  optimizer, online learning, annotation access or evaluator access in the
  proposal/gating programs.

## Frozen target-first branch

1. Resolve the existing OWLv2 name by the exact alias table already used by
   `run_scannet_raw_boxer_clip_vocab_shadow_full100.py`.  Require a target
   alias and Raw-Boxer source score at least `0.40`.
2. At each valid scheduled keyframe, rank target rows by source score
   descending and Raw CSV row ascending; retain at most four rows total.
3. Use the frozen MobileSAM `vit_t` checkpoint with a box-only prompt,
   `multimask_output=True`, and the hypothesis with maximum predicted IoU.
   The predicted-IoU value is diagnostic only and is not an admission gate.
4. Lift the mask with frozen RGB-D geometry: depth `0.10--6.00 m`, one-pixel
   mask-edge erosion, `0.15 m` depth-discontinuity rejection, 2 cm per-view
   voxelization, at least 16 voxels, and at most 2,048 points per view.
5. Rebuild a causal tracker independently for each collapsed target group.
   A current mask AABB may match committed history only when AABB IoU is at
   least `0.10` and center distance is at most `0.50 m`; TTL is ten valid
   keyframes.  Freeze the first three distinct-frame observations.  Same-frame
   observations cannot confirm one another.
6. Fuse the three frozen point fragments on a signed-floor 5 cm grid.  Retain
   support only when at least two views occupy the same or a Chebyshev-adjacent
   voxel.  Require at least 24 supported voxels and at least eight contributed
   voxels from each view.  Fit q02/q98 geometry in the local frame of the
   deterministic Raw-Boxer yaw medoid.  The old single-view world AABB is not
   an active geometry hypothesis.
7. Apply the fixed R15 receipt checks: mean source score at least `0.50`,
   median mask-AABB pairwise IoU at least `0.15`, maximum pairwise center
   distance at most `0.50 m`, first-to-third frame span at least 50, maximum
   camera baseline at least `0.10 m`, maximum viewing-ray span at least 5
   degrees, every fitted extent at least `0.05 m`, and fused center within
   `0.75 m` of the Raw-Boxer evidence medoid.
8. Compare the fused candidate only with the immutable native prefix.  Reject
   when native AABB IoU is at least `0.10` or either directional containment is
   at least `0.50`.
9. Preserve the existing frozen CLIP vocabulary behavior; it may reject a
   suffix candidate but may not change a native label, embedding, score or
   order.  Apply class-agnostic suffix self-NMS at AABB IoU `0.15` or either
   containment `0.25`, then keep at most four births per scene.  Append class
   id `0`, score `1.0`.

## Shadow continuation checks

Before active materialization, the no-GT sidecar must report all of:

- at least 90% valid per-view fragments;
- at least 200 post-fusion, strict-native-novel candidates;
- candidates in at least 70 official scenes;
- measured incremental keyframe latency p95 below 200 ms on an RTX 3090.

Failure stops the branch before an active AP claim.  Passing these engineering
checks authorizes one frozen active official100 evaluation.  AP is not used to
choose among thresholds or suffix geometries in that run.
