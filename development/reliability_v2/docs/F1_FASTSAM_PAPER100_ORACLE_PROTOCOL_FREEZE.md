# F1 FastSAM residual geometry oracle: paper100 protocol freeze

## Status and scope

This document freezes the F1 analysis contract used after the inert F0 run.
F1 is a read-only, offline, GT-assisted geometry oracle. It is not a detector,
does not train or tune anything, does not enable birth, and cannot establish a
deployable AP gain. Its only purpose is to decide whether the already sealed F0
geometry contains enough missed-object capacity to justify implementing a
past-only selector.

This is a protocol freeze, not a claim of blind preregistration: during input
reconnaissance an independent audit process exercised the same fixed matching
formulas before this file was published. No threshold, candidate filter,
geometry, scene, or success criterion may be changed in the production F1 run.

## Frozen inputs

- Scene order: `evaluation/data_util/meta_data/scannetv2_val.txt`, exactly 100
  scenes, SHA-256
  `4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5`.
  It must be exactly the first 100 rows of the sealed F0 full200 list, whose
  SHA-256 is
  `0e7e722d3e93ec4b721f12293a3f1e98ca62d475b42cc8b9d491878a897e9bd1`.
- F0 final receipt:
  `logs/scannet_fastsam_f0_full200_score05/final/F0_FASTSAM_FULL200.json`,
  SHA-256
  `07249ead31ad150cb43d7a35f4c922ac70a8a2f95bcf0fcd24f61f944c1e58a1`.
  It must have `overall_pass=true` and run signature
  `bfc1e14bbbb5507226831efd8b864f69d5ada5dec4b68293d940a45366383286`.
- F0 paper100 sidecars:
  `logs/scannet_fastsam_f0_full200_score05/scenes/<scene>.json`. There must be
  6,817 keyframes, 6,726 successful frames, and 52,299 sealed candidates. The
  canonical ordered hash ledger defined below must equal
  `c2666903a2f8098771d4359d21171fccd8b1df35e38166ef4920251abb94dac7`.
- Frozen native prefix (Cbest: Reliable-View Top-K + Boxer active):
  `results/scannet_t05_boxer_replay_active_score05`, exactly 1,788 boxes. Its
  canonical ordered hash ledger must equal
  `a5566c8b314917d2fa33b69f3e1f7f5372c4e0fe87caf3ab14216e63e6030066`.
- ScanNet GT: `evaluation/data_util/scannet_train_detection_data/<scene>_bbox.npy`,
  exactly 1,433 boxes. Its canonical ordered hash ledger must equal
  `160dec394d87545ee6407f4a734266f65455b61d0b8d4c2701ae70f45f64b287`.
- Axis-alignment metadata:
  `/extra/ZhaoX/scannet_data/scans/<scene>/<scene>.txt`. Its canonical ordered
  hash ledger must equal
  `dead3486b0c6647ae19083673a3451821b88d974a6a9401d06ea252edcbc3e5c`.
- Official constant-score evaluator evidence:
  `upstream_clean/BoxFusion_shallow/evaluation/eval_scannet.py`, SHA-256
  `aea2a72940b7cc53ee273f9f235e2efc848e1994e22da5f439af9751e1e27c27`.

For a ledger, form the scene-ordered JSON value
`[[scene_id, sha256(file_bytes)], ...]`, serialize it as compact ASCII JSON
with separators `(',', ':')`, and SHA-256 the serialized bytes.

All inputs are hashed before and after the audit. The report is create-only and
must not be placed inside a protected input root.

## Frozen candidate geometry

Every `funnel.candidates[]` row from every successful F0 paper100 frame is one
independent candidate. No cross-view clustering, score threshold, confidence
filter, rank cap, size filter, or GT-dependent prefilter is permitted.

The only candidate box is the sealed q02/q98 AABB:

1. validate finite `world_q02`, `world_q98`, `world_center`, and
   `world_extent`;
2. require `q98 > q02`, `center == (q02+q98)/2`, and
   `extent == q98-q02` within the producer's floating-point tolerance;
3. construct all eight corners of the raw ScanNet-world AABB;
4. transform every corner with the scene's `axisAlignment` matrix;
5. take transformed per-axis minima/maxima.

GT boxes are the sealed aligned center/extent arrays converted to min/max.
Native predictions are transformed exactly as in the official evaluator.

## Metrics

All evaluation is class-agnostic. IoU thresholds are exactly 0.15, 0.25, and
0.50, using strict `IoU > threshold` edges.

For each threshold and independently within every scene, report deterministic
maximum-cardinality bipartite matching for:

- native-only boxes versus GT;
- F0-candidate-only boxes versus GT;
- the union of native and F0 candidates versus GT;
- incremental geometry capacity: `union matching - native matching`.

Maximum matching prevents repeated views of one object from being counted more
than once. No matches may cross scene boundaries.

Reproduce the official native constant-score AP first. Every prediction score
is 1.0 and NumPy's default `argsort` tie behavior is retained. The expected AP
points are 31.0130259031, 26.7911284298, and 12.0668518301.

As a secondary diagnostic, construct a separate GT-selected candidate suffix
for each IoU threshold by maximum-matching candidates only to GT instances left
unmatched by the official native greedy evaluation. Append selected rows after
the unchanged native scene rows and evaluate with the same all-1.0 official
ordering. This is a threshold-specific, non-deployable constructive
counterfactual, not a mathematical AP upper bound and not one unified detector
output.

## Frozen decision rule

The requested gain is +10 AP points at each of AP15, AP25, and AP50. With 1,433
GT instances, the necessary geometry-capacity threshold is
`ceil(0.10 * 1433) = 144` additional union matches independently at every IoU.

F1 passes only if all of the following hold:

1. every integrity and baseline-reproduction check passes;
2. `union matching - native matching >= 144` at all three thresholds;
3. the threshold-specific constructive suffix increases official AP by at
   least 10.0 points at all three thresholds.

Failure means this frozen F0 q02/q98 geometry alone cannot support the stated
three-threshold +10 target, so an active FastSAM birth branch must not be
implemented from these results. Passing would only authorize work on a
GT-free, past-only selector; it would not prove that selector can realize the
oracle ceiling.
