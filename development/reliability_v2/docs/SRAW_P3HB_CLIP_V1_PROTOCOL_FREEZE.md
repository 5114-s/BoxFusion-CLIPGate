# SRAW-P3HB-CLIP-v1 protocol freeze

Freeze date: 2026-08-30 (Asia/Shanghai)

This document freezes the only active policy that may be evaluated for the
`SRAW-P3HB-CLIP-v1` paper100 experiment.  The policy must not be changed after
opening the official evaluator output.  ScanNet annotations, oracle-selected
identities, and evaluation metrics are forbidden inputs to the selector.

## Inputs

- The sealed L2 SRAW/F3/F4 identity and geometry receipts.
- The sealed score-0.5 CuTR proposal cache, read only through the current and
  past confirmation frame.
- RGB frames already named in the sealed F4 receipt.
- The existing frozen OpenCLIP ViT-H-14 checkpoint, the unchanged 473-row
  `class_features.pt`, and the unchanged native vocabulary.
- The B05 native prediction only during terminal output reconciliation.

## Causal source and geometry selection

1. Reuse the sealed F3 track identity.  A candidate becomes eligible when its
   third distinct source frame is committed.  Only the first three source
   frames are evidence; later observations cannot change the decision.
2. Each evidence view keeps its original frozen Boxer `HB` geometry.  No
   coordinate averaging, fitting, learning, or target-data calibration is
   allowed.
3. Select the original HB source with maximum mean AABB-envelope IoU to the
   other two HB sources.  Ties use higher HB confidence and then earlier
   source order.

## Frozen geometry admission

All checks must pass:

- three HB confidences are each at least `0.55`;
- median of the three pairwise HB AABB-envelope IoUs is at least `0.25`;
- RMS distance of the three HB centres from their mean is at most `0.25 m`;
- every local HB extent of the selected medoid is at least `0.30 m`;
- selected HB versus same-source H0 normalized centre distance is at most
  `0.50`;
- HB/H0 volume ratio lies in `[0.25, 4.00]`;
- HB/H0 AABB IoU is at least `0.20`, or bidirectional maximum containment is
  at least `0.70`.

These constants are inherited from the already frozen F3/F5/Past3 policies.

## Frozen semantic admission

The three sealed `tight_box_xyxy` RGB crops are scored with the unchanged
native OpenCLIP model, 473-way embedding matrix, and vocabulary.  Existing
ScanNet-compatible rows are only a read-only subset; no new prompt embedding
is created.  All checks must pass:

- at least two of three all-vocabulary top-1 predictions belong to the
  existing target subset;
- at least two target top-1 votes resolve to the same existing alias group;
- median best-target cosine is at least `0.20`;
- median best-target minus best-nontarget cosine is at least `-0.01`.

## Causal novelty and birth control

- At the confirmation frame, compare the candidate with every sealed CuTR
  proposal from current or earlier frames after transforming it to ScanNet
  world coordinates.  Reject at AABB IoU `>=0.10` or either directional
  containment `>=0.50`.
- Compare only with births accepted earlier in causal order.  Reject at AABB
  IoU `>=0.15` or either directional containment `>=0.25`.
- Process equal-frame candidates by geometry support, semantic vote/cosine,
  HB confidence, and stable track/source identity.  Across frames, never
  reorder future candidates ahead of earlier ones.
- Accept at most two births per scene.

## Terminal output reconciliation

The B05 rows remain an exact, unchanged prefix.  At end of stream, suppress
an accepted birth if it overlaps the final native prefix at IoU `>=0.10` or
either directional containment `>=0.50`; this is terminal de-duplication, not
a retroactive birth decision.  Surviving suffix rows use class id `0` and
score `1.0`.  Final suffix self-NMS uses the same `0.15/0.25` thresholds.

## Contracts and interpretation

- Training, fine-tuning, online learning, GT access, annotation access, and
  evaluator access are forbidden during selection and materialization.
- All decisions are current/past-only at their recorded confirmation frame.
- The policy is evaluated once on the official 100 scenes with the official
  constant-score evaluator (`score=1.0`).
- This is an exploratory real active AP test.  The SRAW oracle is not an
  expected active result and no positive AP claim is authorized in advance.
