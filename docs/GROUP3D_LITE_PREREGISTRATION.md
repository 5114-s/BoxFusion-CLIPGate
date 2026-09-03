# Group3D-lite unmatched association preregistration

This document freezes the training-free Group3D-inspired matcher before its
predictions are evaluated. It is not the learned Group3D model.

## Scope and causality

- Run only on real detector keyframes (`frame_id % gap == 0`); never replay a
  stale terminal proposal.
- Consider only current proposals left unmatched by native spatial and
  correspondence association.
- Candidate tracks must exist in the begin-frame past snapshot. A past track
  touched by native association in the current frame is reserved for native
  BoxFusion and cannot be overridden.
- Commit current fragments only after every current-frame match is decided.
  Current-to-current and future-frame matching are forbidden.
- An accepted match keeps the old representative and removes the new
  duplicate. It never changes score, class, vocabulary, or CLIP embedding and
  never creates a new box.
- Any module exception returns the unmodified native association result.

## Frozen voxel rule

Use 5 cm integer voxels. For proposal voxels `Vp` and past-track voxels `Vt`:

```text
I  = |Vp intersection Vt|
J  = I / |Vp union Vt|
Cp = I / |Vp|
Ct = I / |Vt|
D  = 2 I / (|Vp| + |Vt|)
```

Both fragments must contain at least 16 voxels. Accept only if all conditions
hold:

- `I >= 8`
- `J >= 0.10`
- `min(Cp, Ct) >= 0.15`
- `max(Cp, Ct) >= 0.40`

Broad phase requires AABB intersection after expanding each fragment by two
voxels (10 cm). Each proposal considers at most eight nearest candidates.
Conflicts use `D`, then `J`, `I`, centroid distance, and stable track/proposal
ID. Matches must be mutual-best and one-to-one. If a runner-up exists, both
sides require a `D` margin of at least 0.05.

## Fixed resource bounds

- 64 current proposals per keyframe, chosen by frozen detector score with
  proposal ID as deterministic tie-break.
- 8 candidates per proposal.
- 512 voxels per view, 5 views per track, 1024 union voxels per track.
- 1024 cached tracks. LRU eviction drops only the auxiliary cache and never a
  native prediction.
- CPU NumPy implementation; missing, non-finite, or insufficient fragments
  abstain.

Raw and cleaned arms must use the identical 64-proposal selection. Fragment
voxel count must not influence arm membership.

## Factorial attribution

| Arm | Definition | Attribution |
| --- | --- | --- |
| E2 | chosen result of the gate x Top-K 2x2 | formal base |
| Sshadow | E2 plus SMOV extraction only | native output identity/runtime |
| Graw | E2 plus the matcher using an uncleaned depth crop | Group association effect |
| Gclean | same matcher and thresholds using SMOV-clean fragments | SMOV cleanup increment |

- `Graw - E2` estimates Group3D-lite association.
- `Gclean - Graw` estimates SMOV cleanup independently.
- `Gclean - E2` estimates their combined effect.

## Evaluation gates

The fixed smoke scenes are official-list indices 0/20/40/60/80:

```text
scene0568_00
scene0193_01
scene0695_01
scene0064_00
scene0207_02
```

They were selected by list position without GT inspection. The five-scene
stage checks correctness and gross regressions; it is not a final accuracy
claim. Advancement requires no crash, prefix-causality parity, all resource
bounds, FPS ratio at least 0.95, AP25/AP50 nonnegative, AP15 at least -0.10 AP,
and mean delta at least +0.10 AP. Fewer than five accepted matches is reported
as underpowered and does not authorize threshold tuning.

Full100 retention under the official in-memory `score=1.0` evaluator requires:

- positive AP25 and AP50 deltas;
- AP15 delta at least -0.10 AP and mean delta at least +0.20 AP;
- for AP25/AP50, at least one paired-bootstrap 95% lower bound nonnegative and
  the other at least 90% positive resamples;
- paired full-pipeline FPS ratio at least 0.95 and association p95 no more than
  5 ms per keyframe;
- all causality, bounded-memory, training-free, and native-field audits pass.

Because the official evaluator forces every score to 1.0, no claim may rely
on a low-score suffix to protect AP.
