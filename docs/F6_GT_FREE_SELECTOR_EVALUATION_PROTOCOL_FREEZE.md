# F6 multi-view selector — one-shot paper100 evaluation freeze

Frozen: 2026-08-29 (Asia/Shanghai), before any F6 paper100 execution or F6
comparison with GT.

Evaluation protocol ID:
`F6-GT-FREE-SELECTOR-ONE-SHOT-EVALUATION-PAPER100`.

## Boundary

Only a complete F6 no-GT merge whose frozen retain gate passes may be
evaluated.  The evaluator is a separate process.  F6 itself never reads GT,
native predictions or evaluator output.  The evaluation is a geometry
capacity test, not actual active-birth AP: F6 produces no detection and the
native constant-score output remains unchanged.

The evaluator pins and re-hashes the F4/F6 receipts, all scene sidecars, the
paper100 scene list, ScanNet GT/axis alignment, Cbest predictions and official
evaluation source.  It must reproduce exactly 100 scenes, 52,299 sources,
1,788 native boxes, 1,433 GT boxes and the constant-score native AP
`31.0130259031 / 26.7911284298 / 12.0668518301`.

## One geometry per source

Every selected geometry must be byte-equivalent to exactly one F4
`H0/HL/HLG/HB` hypothesis with complete lineage.  AABB corners and true HB OBB
corners are transformed by scene alignment before taking aligned min/max.
Alternatives are never stacked.  Strict IoU comparisons are `>0.15`, `>0.25`
and `>0.50`.

At each threshold report maximum matching for native, selected geometries and
their union; extra union matches; results split by selected hypothesis and by
base/switch; and retained fraction of frozen F4 G4 capacity.  Also report a
constructive suffix restricted to official-native-greedy-unmatched GT, one
selected source per GT, frozen source order and formal score 1.0.  This suffix
is GT-selected, nondeployable and must never be called actual F6 AP.

## Fixed decision

F6 passes only if its no-GT merge passed and, independently at all three IoU
thresholds:

1. selected geometry plus native adds at least 144 maximum-matching GT
   instances; and
2. the GT-selected constructive suffix improves native AP by at least 10.0
   points.

Failure yields `discard_f6_multiview_selector_for_plus10_route`, with no
paper100 threshold/rule revision.  Passing yields
`retain_f6_authorize_f7_high_precision_birth_shadow_only`; it does not
authorize an active birth, terminal mutation or deployment.  A final claim
still requires the unchanged protocol on an independent held-out scene set.
