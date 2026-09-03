# PUF-geometry-shadow preregistration

Date frozen: 2026-08-23 (Asia/Shanghai), before inspecting PUF-shadow AP.

Pre-AP implementation-audit amendment: the exact paper-rule directive below
is retained as a diagnostic, but it is not by itself eligible for an active
BoxFusion association.  This amendment was frozen before running inference or
inspecting any PUF-shadow AP.

## Scope

This stage is a training-free, output-inert adaptation of PUF's voxel node
association to BoxFusion detection.  It is **not** the full PUF scene-graph
model.  It uses no ScanNet labels, fitted weights, validation search, class
co-occurrence prior, Dirichlet semantic update, relationship update, or birth.
CuTR, Reliable-View Top-K and CLIP remain frozen and unchanged.

The online input is the existing Gclean stream: SMOV-clean signed 5 cm voxel
fragments from native-unmatched current proposals and a bounded
begin-keyframe-past voxel memory.  Tracks already reserved by native
association are ineligible.  Current fragments enter memory only after the
current query, so current-to-current and future association are impossible.

## Frozen probability rule

For current proposal voxel set `Vp` and each eligible historical candidate
`Vk`, the geometry-only likelihood is the asymmetric containment used by the
PUF voxel backend:

```text
Lk = |Vp intersection Vk| / |Vp|
Z = 0.4 + sum_k Lk
beta_k = Lk / Z
beta_null = 0.4 / Z
```

`L_sem` is fixed to one because changing or re-accumulating CLIP semantics is
outside this experiment.  The public PUF default `0.4` is frozen before this
run and interpreted only as a null/unmatched mass.  It is not configurable and
cannot create a box.

For each valid proposal, normalization includes every positive-intersection
candidate surviving the already-frozen bounded broad phase: a two-voxel AABB
expansion followed by at most eight nearest tracks, with stable track ID as a
tie break.  Group3D's hard intersection, IoU, containment, mutual-best and
runner-up gates are not applied to this probability table.

The preregistered paper-rule directive is:

```text
if beta_null <= 0.5:
    associate counterfactually with argmax_k beta_k
else:
    remain unmatched
```

Exact likelihood ties select the smallest stable track ID.  No posterior,
entropy, margin, detector-score or GT threshold is added after viewing AP.
Normalized association entropy is diagnostic only and is not PUF's label
Dirichlet entropy.

For a mechanically valid BoxFusion counterfactual, a paper directive is
`active-safe` only when its selected track is also strictly more probable than
the null state (`margin > 0`) and no other proposal in the same keyframe selects
that historical track.  Every member of a same-track conflict group is
excluded; no winner is chosen.  This deterministic safety rule introduces no
tunable constant.  Both the raw paper directives and the smaller active-safe
set are serialized, and only the active-safe set may be materialized for AP.

## Bounds and failure behavior

- At most 64 current proposals, 8 candidates per proposal and 512 candidate
  pairs per keyframe.
- At most 1,024 past tracks, 5 retained views per track, 512 voxels per view
  and 1,024 union voxels per track.
- All IDs, counts, likelihoods and probabilities must be finite and valid;
  each probability row must sum to one with absolute error at most `1e-12`,
  recorded per row and as a batch maximum.
- Structural error, cap violation or invalid trace fails open to no directive.
- A fail-open result from either the frozen Gclean hard matcher or the PUF
  evidence/probability path fails the entire PUF keyframe open.
- PUF-shadow cannot mutate native rows, geometry, class, score, order, CLIP,
  BoxFusion state or random-number state.

## Frozen preflight protocol

- Scenes, in order: `scene0568_00`, `scene0606_01`, `scene0377_02`.
- Sealed proposal namespace:
  `scannet-graw-e2-score05-preflight3-v3-r1`.
- Producer fingerprint:
  `457c997631cd71a83b6480a6e45e103e273ac5ed2d1488252790549c2e2b3504`.
- CuTR `score_thresh=0.5`, gap 25, appearance gate disabled,
  Reliable-View Top-K3 enabled.
- Headline AP uses the official evaluator's in-memory constant score `1.0`,
  matching the formal B05/T05 protocol.  Native disk-score AP is reported
  separately as a diagnostic and must not be mixed with the headline result.

## Counterfactual and advancement gate

Each active-safe directive records the current native track and the proposed
past track.
The terminal audit follows only strictly later native aliases.  Cases already
made identical by native association, or where either side is dropped, are
no-ops.  Only a pair that survives as two distinct terminal rows removes the
candidate row.  Retained row order, geometry, class and score stay byte-exact.
This authorizes only duplicate-suppression counterfactual evaluation, not a
live fusion override.

The three scenes first gate mechanics and gross regressions.  PUF-active and a
full100 run are prohibited unless all traces are causal and valid, shadow
outputs are noninterfering, probability invariants pass, and the constant-score
counterfactual has AP25 and AP50 deltas greater than zero, AP15 delta at least
-0.10 point, and mean AP15/AP25/AP50 delta at least +0.20 point.  Fewer than
five terminal-actionable directives is reported as underpowered and cannot
authorize active mode or threshold tuning.

The added PUF evidence extraction plus probability calculation must have p95
latency no greater than 2 ms per processed keyframe.  Probability-only latency
is reported separately.  The total SMOV+Gclean+PUF observer cost is also
reported both per keyframe and amortized by the fixed gap of 25.

## Interpretation boundary

PUF reports large gains for online 3D scene-graph relationship recall, not
ScanNet 3D detection AP.  Its complete model consumes soft 2D scene-graph
distributions, and one experiment uses a relationship prior computed from
training annotations.  Neither component is used here, so those published
gains are not an expected AP gain for this module.
