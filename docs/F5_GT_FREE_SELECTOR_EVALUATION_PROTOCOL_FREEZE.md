# F5 GT-free selector — one-shot paper100 evaluation freeze

Frozen: 2026-08-29 (Asia/Shanghai), before the F5 paper100 selector replay and
before any F5 row was compared with GT.

Evaluation protocol ID:
`F5-GT-FREE-SELECTOR-ONE-SHOT-EVALUATION-PAPER100`.

## Purpose and boundary

This evaluation asks whether the single geometry selected by the frozen F5
rule retains enough of the sealed F4 geometry capacity to support the stated
`+10 AP-point` research target.  The F5 rule itself remains GT-free.  GT is
opened only by this separate evaluator after a complete passing no-GT F5
merge exists.

F5 does not select which source becomes a detection.  Consequently this
evaluation is a geometry-capacity test, not an active-birth or deployable-AP
test.  Any constructive suffix below is selected with GT and must be labelled
oracle-only.  Native predictions, scores, semantics, CLIP, source order and
files remain unchanged.

## Frozen inputs

- the exact passing F4 paper100 merge and 100 F4 scene sidecars;
- the exact create-only F5 paper100 merge and 100 F5 scene sidecars;
- the same paper100 scene list, ScanNet boxes, axis-alignment files, official
  evaluator and constant-score Cbest prefix used by F1--F4;
- the historical sealed F4 oracle only as an aggregate reproduction anchor,
  never as an input feature or selector label.

The evaluator must re-hash all inputs before and after and refuse a partial,
failed or non-deterministic F5 merge.  It must reproduce exactly 100 scenes,
52,299 sources, 1,788 native boxes and 1,433 GT boxes.

## One geometry per source

For every F5 source, validate the complete F4 identity and input-hypothesis
hash ledger.  The F5 selected hypothesis must be exactly one of
`H0/HL/HLG/HB`, and its geometry must be an exact copy of that F4 hypothesis.
Alternative hypotheses are never stacked.

- H0/HL/HLG: transform all eight world-AABB corners with the scene alignment,
  then take aligned min/max.
- HB: transform the eight sealed world-OBB corners first, then take aligned
  min/max.  Axis-aligning an HB envelope before the transform is forbidden.

Strict IoU comparisons are `>0.15`, `>0.25` and `>0.50`.

## Required reports

At each threshold, report:

1. the official constant-score native baseline;
2. maximum matching for native, all F5-selected geometries, and their union;
3. additional union matches over native;
4. results split by selected H0, HL, HLG and HB;
5. HB-selected source count, matched count and match fraction, both against
   all GT and against official-native-greedy-unmatched GT;
6. the historical F4 G4 capacity and the fraction retained by F5;
7. a constructive suffix restricted to official-native-greedy-unmatched GT,
   with maximum matching, one selected geometry per source, frozen source
   order and formal score 1.0.

The constructive suffix is explicitly GT-selected and nondeployable.  Its AP
must not be described as the actual F5 AP.

## Fixed decision

F5 geometry selection passes only if:

- no-GT F5 integrity, causality, determinism and runtime all passed;
- the native baseline reproduces
  `31.0130259031 / 26.7911284298 / 12.0668518301`;
- F5-selected geometry adds at least 144 native-union matches independently
  at all three thresholds; and
- its GT-selected constructive suffix improves the native AP by at least
  10.0 points independently at all three thresholds.

A pass yields
`retain_f5_authorize_new_preregistered_birth_confirmation_shadow_only`.
It does not authorize active birth.  A failure yields
`discard_f5_geometry_selector_for_plus10_route`; no F5 threshold or tie rule
may be changed and rerun on paper100.

