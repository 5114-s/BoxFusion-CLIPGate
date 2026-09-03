# PUF arbitration-lite ownership shadow

PUF-lite scores each CuTR proposal independently.  Two duplicate proposals can
therefore recommend the same historical track.  That is not automatically an
error: native BoxFusion may legitimately merge several observations into one
track in the same NMS pass.

This module implements conservative **directive ownership**, not one-to-one
object matching.  It never changes native association in the current stage.

## Frozen rule

For a PUF track decision, let `p` be the selected track posterior and

```text
competitor = max(beta_birth, every other track posterior)
margin = p - competitor
```

A non-conflicting track recommendation is eligible only when:

```text
p >= 0.70
margin >= 0.20
candidate source == QIM
```

For proposals selecting the same historical track, sort contenders by:

```text
-p, -margin, -likelihood, -overlap_support,
-shared_key_fraction, QIM source/rank, proposal_id
```

The first contender becomes the sole owner only when it passes the row gate
and exceeds the second contender's posterior by at least `0.10`.  Otherwise
the entire group abstains.  Every non-owner is `native_fallback`; it is never
rerouted to an alternative track, changed to birth, or suppressed.

A raw PUF birth is eligible only when `beta_birth >= 0.70` and its margin over
the strongest track is at least `0.20`.  All thresholds are frozen in config
and rejected if changed while the observer is enabled.

## Causal and semantic contract

The arbiter receives only an immutable `PUFQueryBatch`, immediately after PUF
has frozen its pre-association posteriors.  It receives no boxes, image,
detector score, CLIP feature, category, BoxManager, merge event, native target,
or ground truth at query time.

Native targets are supplied only after unmodified spatial/correspondence NMS
and are used for diagnostics.  Observation updates counters only; changing a
native label cannot change any later arbitration decision.

The summary explicitly asserts:

- `observer_only=true`, `active_authorized=false`;
- `training_free=true`, `online_update=false`, `causal=true`;
- no semantic, detector-score, or ground-truth access;
- `reassigns_losers=false`, `suppresses_proposals=false`;
- zero duplicate selected-track directives.

## Native conflict-group metrics

The union-find target trace may produce more than one native-positive proposal
for a contested track.  Metrics therefore distinguish:

- fully resolved groups with one native-positive proposal;
- fully resolved groups with multiple native-positive proposals;
- native-unsupported groups;
- groups containing any unresolved member;
- whether an emitted owner belongs to the native-positive set;
- how often a deferred loser is itself native-positive.

Any unresolved member removes the whole group from owner precision.  An
all-abstain group has no owner and contributes no loser-precision denominator.
This avoids treating legal many-proposal-to-one-track merges as arbitration
errors.

## Tests and benchmark

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/admin1/miniconda3/envs/boxfusion2/bin/python -m pytest -q \
  tests/test_puf_arbitration_lite.py \
  tests/test_audit_moon_qim_puf_paired.py

/home/admin1/miniconda3/envs/boxfusion2/bin/python \
  tools/benchmark_puf_arbitration_lite.py \
  --proposals 64 --group-size 2
```

At 64 proposals arranged into two-member conflict groups, CPU query p95 is
about `0.56 ms/keyframe`.  At the hard 256-proposal cap it is about
`2.25 ms/keyframe`, or `0.090 ms/input frame` at BoxFusion gap 25.  Real scene
latency, rather than the synthetic cap, is the activation criterion.

## Single-scene result

On `scene0277_00`, seed 0, GPU 1, real CuTR and no proposal cache:

- final output is byte-identical to the warm control;
- observer/control FPS is `34.12/33.83 = 1.009x`;
- arbitration query p95 is `0.045 ms/keyframe` and wrapper overhead is
  `0.00118 ms/input frame`;
- 2 raw conflict groups contain 4 proposals;
- both groups are native multi-positive and both abstain because no proposal
  passes the owner gate;
- zero duplicate directives and zero wrong/false track/birth directives;
- 2 non-conflicting track and 10 birth directives are all native-consistent.

This validates the shadow mechanics, but provides zero evaluable conflict
owners.  It cannot authorize active ownership.  The pre-registered fixed-10
run must contain at least 50 evaluable owners for the planned activation gate;
otherwise the module remains shadow regardless of its precision.

## Frozen fixed-10 result

The seed-0 fixed-10 run used real CuTR proposals, one physical GPU, the same
model assets, and separate unmodified-control and observer passes.  Both passes
completed all 10 scenes and produced valid prediction files and logs.

The activation gate **failed**, so arbitration remains shadow-only:

- one native-relative false-birth recommendation occurred among 388 selected
  high-confidence directives (`scene0598_01`, frame 175, proposal 18);
- only 13 conflict groups emitted an owner, below the pre-registered minimum
  of 50 evaluable owners;
- minimum QIM Recall@K was `0.987406`, and minimum PUF post-fallback coverage
  was `0.987245`;
- maximum PUF query p95 was `4.113 ms/keyframe`, above the `2 ms` gate;
- combined QIM + PUF + arbitration wrapper overhead was
  `0.119289 ms/input frame`, above the `0.10 ms` gate;
- arbitration itself remained small at about `0.002312 ms/input frame`, with
  zero duplicate selected tracks, zero invalid source rows, and zero false
  track directives.

Only 3/10 scenes were byte-identical across the two independent GPU processes.
The other seven preserved row count, labels, and scores, while box corners
differed by at most `2.6047e-05 m`.  This cross-process numerical drift does
not authorize relaxing the strict gate.  A future run should additionally
hash native state immediately before and after each observer call in the same
process, so observer mutation is tested independently from GPU
nondeterminism.

Observer/control FPS ratios ranged from `1.205` to `2.575`, but the control-all
then observer-all launch order introduced a strong cold/warm-cache bias.  These
numbers only show that no slowdown was observed; they must not be reported as
an acceleration caused by the shadow modules.  On this hardware, the control
itself ran at `12.39--29.18 FPS`, so the experiment also does not establish an
absolute 20-FPS guarantee.

The next safe experiment is an independent MV3DIS-inspired depth-consistency
shadow observer.  It should diagnose fusion-view weights and birth vetoes
without activating PUF, changing native association, or touching CLIP
semantics.
