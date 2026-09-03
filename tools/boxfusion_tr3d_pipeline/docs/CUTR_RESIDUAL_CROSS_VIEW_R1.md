# CuTR residual cross-view R1 (pre-registered shadow)

This document freezes the R1 decision rule before its first end-to-end
evaluation. R1 is an observer-only confirmation layer over the S0 low-score
CuTR residual tracker. It adds no model, training, online optimization, CLIP
call, ground-truth access, or future-frame access.

## Frozen evidence

Each accepted residual row may contribute one node containing its true CuTR
frame id, raw row id, copied 256-D CuTR decoder descriptor, camera pose, raw 2D
box, and a 16--64 point metric-depth guide. The descriptor is the existing
CuTR decoder hidden state, not CLIP; it is explicitly L2-normalized before
cosine comparison. Historical depth images are never retained.

An edge is evaluated only from an earlier committed guide into a later true
CuTR frame. The current guide is committed after all queries and can never
self-confirm. An edge is supporting exactly when all of these fixed conditions
hold:

- camera translation is greater than `0.8 m` **or** relative SO(3) rotation is
  greater than `30 degrees` (the released BoxFusion association gaps);
- normalized CuTR descriptor cosine is at least `0.80`;
- MV3DIS-style relative-depth tolerance is `alpha=0.05`;
- forward visibility `Vf > 0.30`;
- proposal-conditioned box visibility `Vb > 0.90`.

`Df`, `Db`, affinity, camera-to-object ray angle, and raw descriptor cosine are
reported as diagnostics but introduce no additional threshold. `Ivis` inside
`Vf` already requires `abs(z-d) < 0.05*d`; adding a second soft-depth threshold
would be an unregistered hyperparameter.

R1 admits an S0 terminal candidate only if the supporting-edge graph has a
weakly connected component containing at least three distinct true CuTR frames
and at least two supporting edges. This is ANDed with every existing S0
stability, size, native-novelty, self-NMS, and output-cap gate. Missing
descriptor/pose/guide/depth, invalid sensor data, an incomplete projection, or
a point-budget exhaustion abstains and cannot form an edge.

## Bounds and isolation

- residual rows considered per keyframe: at most 64, ranked by
  `(-score, raw_row_id)` before descriptor/guide GPU-to-CPU copies;
- historical nodes per residual track: at most 5;
- points per guide: at most 64;
- historical-to-current projection budget: 8192 points per true CuTR keyframe;
- native CuTR, lifting, CLIP, BoxManager, BoxFusion, post-process, and export
  paths remain unchanged;
- R1 output is a counterfactual subset of S0 candidates and is never appended
  by the live demo.

The 8192-point budget is scheduled deterministically. Any row whose required
history cannot be evaluated completely before exhaustion abstains; partial
edges are not committed. Retired unconfirmed histories are reclaimed and any
confirmed receipt archive is bounded. Capacity loss makes the supplemental
audit incomplete but never changes or stops native inference.

## Pre-registered gates

The first run is `scene0598_01`, used only after the source/config hashes are
frozen. It is a falsification smoke, not a threshold-fitting scene.

- native shadow output: labels and scores exact; geometry within `5e-5 m` of a
  same-code warm control;
- end-to-end FPS retention: at least `0.95`;
- R1 wrapper p95: at most `25 ms` per CuTR keyframe and at most `1 ms` amortized
  per input frame;
- audit/capacity/identity violations: zero;
- create-only append counterfactual: AP15/AP25/AP50 and recall must not decrease;
- any appended false positive at IoU 0.15 keeps R1 unauthorized;
- a zero-candidate result is safe but is not evidence of an accuracy gain.

Only a passing smoke permits a separately frozen fixed-10 shadow evaluation.
Neither the cosine threshold nor any geometric/depth threshold may be lowered
after observing `scene0598_01`; a failed or empty result remains shadow-only.

## Measured result (2026-08-21)

R1 remains shadow-only. On `scene0598_01`, it rejected all six S0 candidates.
The official AP15/AP25/AP50 therefore remained
`0.277389/0.277389/0.030303`; this prevented the known S0 precision loss but
did not add a true positive. The final observer ran at 32.00 FPS versus 31.47
FPS for the current-code control. Its wrapper p95 was 11.06 ms per CuTR
keyframe (0.44 ms per input frame at gap 25). Labels and scores matched the
control exactly, but native particle-search geometry differed by 2.89 mm
across runs, beyond the registered 0.05 mm cross-run tolerance. The in-process
identity guard nevertheless observed zero mutation of any native CuTR field.

The pre-registered fixed-10 shadow processed 626 CuTR keyframes and 13,632
accepted residual rows. S0 produced 52 terminal candidates; R1 retained only
one (`scene0496_00`, track 522). That candidate was a false positive at IoU
0.15; its maximum GT IoU was 0.079862. On the six audit-complete scenes,
create-only evaluation was:

| output | AP15 | AP25 | AP50 | P@15 | P@25 | R@15 | R@25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| native | 0.447700 | 0.438406 | 0.249842 | 0.533333 | 0.520000 | 0.588235 | 0.573529 |
| native + R1 | 0.447700 | 0.438406 | 0.249842 | 0.526316 | 0.513158 | 0.588235 | 0.573529 |

Four of ten scenes were not auditable because the frozen 64-row S0 cap
dropped 378 low-score proposals, and `scene0568_00` exceeded the registered
wrapper-p95 limit (26.53 ms versus 25 ms). There were no R1 projection errors
or projection-budget abstentions. Consequently R1 demonstrates an aggressive
proposal veto (52 to 1), but **no accuracy gain**, and neither activation
nor threshold retuning on these validation scenes is authorized. The complete
machine-readable report is
`reports/cutr_residual_birth/fixed10_r1_v1.json`.
