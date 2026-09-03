# PUF-lite training-free causal shadow

PUF-lite is the second step after Moon-QIM-lite.  It keeps PUF's explicit
birth state and per-observation probability normalization, but removes every
learned or semantic component.  In this stage it is an observer only: native
BoxFusion remains the sole association path.

## What is and is not borrowed

From PUF, this implementation retains

\[
Z_i=\lambda_{birth}+\sum_{k\in C_i} L_{ik},\qquad
\beta_{ik}=L_{ik}/Z_i,\qquad
\beta_i^{birth}=\lambda_{birth}/Z_i.
\]

It uses PUF's strict decision rule: predict birth only when
`beta_birth > 0.5`; at exactly `0.5`, associate with the maximum-likelihood
track.  The frozen literature value is `lambda_birth=0.4`.  This value must
not be retuned on BoxFusion validation or evaluation ground truth.

PUF's semantic JSD likelihood, Dirichlet node/edge updates, relation prior,
and train-count prior are absent.  PUF-lite never receives an image, category,
CLIP embedding, detector score, native association result, or ground truth at
query time.  The geometry likelihood below is a BoxFusion-specific
training-free approximation, not PUF's original spatial likelihood.

For a proposal and historical track AABB, let `r` be intersection volume over
proposal volume, `u` be AABB IoU, `q` be Moon-QIM shared-key fraction, and

\[
g_{ov}=\sqrt{ru},\qquad
g_{ctr}=\exp\left[-\frac12\sum_a
\left(\frac{c_{ia}-c_{ka}}
{0.5(s_{ia}+s_{ka})/2+0.05\text{ m}}\right)^2\right].
\]

The fixed likelihood is

\[
L_{ik}=\operatorname{clip}
\left(g_{ov}+(1-g_{ov})qg_{ctr},0,1\right).
\]

This suppresses a large track merely containing a small proposal through the
IoU factor, while allowing a small lifting displacement to recover through
the sparse-key and center terms.

## Causal execution

For every real CuTR keyframe:

1. CuTR proposals are transformed into metric world coordinates.
2. Moon-QIM queries only tracks committed through the previous keyframe.
3. PUF-lite maps stable track IDs to the frozen pre-association rows and scores
   the QIM unique Top-3.
4. If their total likelihood is below `0.4`, a vectorized fallback scans the
   previous active pool, capped at 1024 tracks, and retains only its Top-3.
5. PUF posteriors and the birth decision are frozen.
6. Unmodified BoxFusion spatial/correspondence association and fusion run.
7. Native association is used only to record shadow diagnostics; it cannot
   update a likelihood, threshold, or parameter.

QIM candidates that claim to be active but are absent from the causal ID
registry, candidates whose recorded IoU/distance disagrees with the frozen
track snapshot, non-finite boxes, non-positive extents, over-capacity history,
or non-normalized probabilities yield `invalid/no recommendation`, never an
artificial birth.  Multiple simultaneous proposals selecting one track are
marked non-actionable conflicts.  All inputs passed to PUF are copied arrays;
no mutable BoxFusion object is exposed.

## Metrics and activation gate

The summary deliberately separates:

- QIM target membership at Top-3;
- positive-likelihood support after fallback;
- PUF Top-1 agreement with a unique native history target;
- set-valued diagnostics for ambiguous multi-history targets;
- native birth precision and recall;
- invalid rows, same-track conflicts, fallback triggers/rescues, NLL/Brier,
  normalization error, and module timing.

Agreement with native association is a diagnostic, not an AP result.  Shadow
mode must produce identical output and therefore cannot improve AP.  Before
any active override, run a frozen fixed-10 scene set and require at least:

- byte-identical output, zero invalid/non-finite rows, and no GT/semantic
  access;
- post-fallback positive target support at least 99.9%;
- unique-target Top-1 agreement at least 95%;
- same-track conflict rate at most 0.5%;
- observer/control FPS ratio at least 0.95;
- PUF keyframe p95 at most 2 ms on the declared hardware.

The first active variant must be selective and fall back to native BoxFusion
unless the winning posterior is at least 0.70 with a margin of at least 0.20.

## Reproducible checks

Unit and integration regression:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/admin1/miniconda3/envs/boxfusion2/bin/python -m pytest -q \
  tests/test_puf_lite.py tests/test_moon_qim_lite.py \
  tests/test_audit_moon_qim_puf_paired.py
```

CPU microbenchmark:

```bash
/home/admin1/miniconda3/envs/boxfusion2/bin/python \
  tools/benchmark_puf_lite.py --tracks 128 --proposals 32
```

Use `config/scannet_qim_puf_shadow.yaml` for the real stream.  The paired
audit tool accepts `--require-puf` and checks prediction bytes, safety flags,
QIM recall, post-fallback support, FPS, and combined observer overhead.

## Current real-stream result

On `scene0277_00`, seed 0, GPU 1, real CuTR inference, no proposal cache, and
the repository-root working directory:

- the eight final boxes have exactly the control SHA-256
  `c66d67649f14ef359a344102dbd414ac7c99c2dd70d5cee0c8499804ffdd89e3`;
- observer/control FPS is `34.30/33.83 = 1.014x` (runtime noise, not a speedup);
- QIM+PUF wrapper overhead is `0.0585 ms/input frame`, of which PUF is
  `0.0199 ms/input frame`;
- PUF query p95 is `0.936 ms/keyframe`, with zero invalid or non-finite rows;
- unique history Top-1 is `45/45`, native birth precision/recall is `10/10`,
  and two ambiguous target sets are both supported correctly;
- 4 of 57 proposals form same-track conflicts (`7.02%`).

The last item fails the proposed active gate, so this result authorizes the
shadow implementation only.  It does not authorize PUF-lite to replace native
association.  The machine-readable audit is
`reports/puf_lite/smoke_scene0277_00_v2.json`.

The follow-up conservative ownership observer is documented in
`docs/PUF_ARBITRATION_LITE.md`.  It treats conflicting losers as native
fallbacks rather than births or alternative associations.

The dense synthetic fallback is intentionally reported separately: with 10
proposals and the maximum 1024 tracks, p95 is about 6.30 ms/keyframe
(`0.252 ms/input frame` at gap 25).  A dense fixed-10 real test is therefore
required before making a general real-time claim.
