# CA-1M C3: native final-OBB quality observer

## Purpose

The earlier P2 experiment applied a ScanNet-trained B6 head to only the
CA-1M rows reached by an evolving YOLOE/object-memory track.  It covered only
`199/674 = 29.53%` of the frozen fixed-ten predictions and its score-only
counterfactual reduced AP.  C3 tests a different contract before any CA-1M
training is authorized:

```text
frozen CA-1M P1 (score=0.4 + Selective Boxer G0)
-> cache every actual G0 keyframe depth/K/pose
-> query every final yaw OBB against those keyframes
-> stable top-K depth/support/occlusion/free-space evidence
-> write features only; preserve the prediction exactly
```

C3 does not start YOLOE, masked CLIP, object memory, ScanNet B6, or the old
`online_refinement` controller.  It is final-box-centric and is designed to
make diagnostic row mapping independent of proposal-track eligibility.

## Frozen contract

- dataset: CA-1M fixed ten validation scenes;
- detector threshold: real detector score, `score_thresh=0.4`;
- lifting/fusion: frozen P1 Selective Boxer G0;
- CuTR proposals: immutable cache replay namespace
  `ca1m-score04-gap20-c0-v2`;
- geometry: world yaw OBB is preserved when projecting and classifying depth;
- views: stable top 5, depth pixel stride 4;
- mutations: geometry, score, label, row count, and row order are all disabled;
- supervision: no CA-1M GT is read by the observer or identity audit.

The authoritative identity comparison is not the historical P1 output.  The
runner writes canonical create-only predictions immediately before and after
the native observer in the same process.  Those two pickle files must have
the same SHA256 and must be semantically identical row by row.  Historical
P1 remains a replay/runtime reference because independent GPU fusion can
have small floating-point geometry drift.

The diagnostic audit requires:

- `result_indices == arange(N)` for every scene, including `N=0`;
- diagnostic corners and scores exactly match the post-observer prediction;
- the `detector_score` feature exactly matches the source score;
- all features are finite and use one common ordered schema;
- mapping coverage is 100%;
- `observer_only=true`, `mutation_enabled=false`, and `applied_count=0`;
- Selective-Boxer deterministic fields agree with frozen P1.

The hardened v2 audit additionally recomputes rather than trusts redundant
NPZ fields:

- `aggregate_depth_counts == sum(per_view_depth_counts, view)`;
- per-view and aggregate support/occlusion/free-space/invalid fractions;
- aggregate selected-view count and sampled-ray count;
- `projectable` and `valid_evidence` from the underlying Top-K/count arrays;
- each yaw box from the saved eight world corners;
- all 14 ordered quality features from score, yaw box, projected area, and
  depth counts.

It records every row's selected, sampled, and classified view counts plus
total/classified ray samples.  Per-scene and aggregate reports include
distributions and count strata, so 100% row mapping cannot hide boxes with
zero classified depth samples.  Runtime reporting contains both
frame-weighted FPS and the minimum individual-scene FPS.

`valid_evidence_coverage` is separate from mapping coverage.  A mapped final
box may legitimately have no valid projected depth rays; the observer must
record it rather than silently drop the row.

## Run

Preflight only (does not create formal result/diagnostic/report roots):

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
BOXFUSION_CA1M_C3_PREFLIGHT_ONLY=1 \
bash scripts/run_ca1m_c3_native_b6_observer.sh 0,1
```

Fixed-ten observer, audit, and standard CA-1M evaluation:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/run_ca1m_c3_native_b6_observer.sh 0,1
```

The runner is create-only and refuses partial or existing namespaces.  After
diagnosing a failed attempt, select a new tag instead of overwriting it:

```bash
BOXFUSION_CA1M_C3_RUN_TAG=ca1m_c3_native_b6_observer_fixed10_v2 \
bash scripts/run_ca1m_c3_native_b6_observer.sh 0,1
```

The standard AP must equal the same-run P1 anchor because C3 is observer-only.
The formal outputs are:

- `reports/ca1m_port/<tag>/identity_audit.json` — identity, 100% row mapping,
  valid-depth coverage, feature schema, and paired runtime;
- `logs/ca1m_port/<tag>/eval.log` — unchanged AP15/AP25/AP50;
- `diagnostics/ca1m_port/<tag>/native_b6/` — create-only per-scene NPZ files.

## Decision before training

Fixed-ten validation is a plumbing/coverage test, not a training or parameter
selection set.  Proceed to a new train-only CA-1M download/conversion stage
only when:

1. all ten same-run prediction hashes are identical;
2. final-row mapping coverage is exactly 100%;
3. valid depth evidence coverage is high enough to support a general quality
   head (use 80% as an engineering warning threshold, not a tuned gate);
4. features have finite, non-degenerate distributions;
5. observer kernel time and end-to-end cost are recorded separately.

If these checks pass, train a class-agnostic, yaw-aware quality head on a
deterministic CA-1M **train-only** subset with scene-level train/development/
calibration splits.  Do not use canonical103, derived107, fixed10, or any
validation GT for fitting or threshold selection.  CLIP and the proposal
semantics remain frozen.  The first active validation run is allowed only
after a train-only gate is frozen.

The report deliberately separates two decisions:

- `engineering_identity.ok` means the same-run outputs and diagnostic
  redundancies passed the fail-closed audit;
- `train_readiness.authorized` is always `false` for this fixed-ten
  validation-only protocol, even if every engineering prerequisite passes.

Consequently, `ok: true` is not permission to fit a model, select a feature,
choose a threshold, calibrate a score, or activate a validation result.  A
separate audited CA-1M train-only collection is mandatory.

## Fixed-ten post-run audit result

The completed v1 observer artifacts contain 674 final predictions.  The
hardened audit must be run create-only to a new report path; it must not
replace the immutable formal report:

```bash
python tools/audit_ca1m_c3_native_b6_observer.py \
  --scene-list evaluation/data_util/meta_data/ca1m_val_ablation10_even.txt \
  --anchor-root results/ca1m_port/ca1m_c3_native_b6_observer_fixed10_v1_same_run_anchor \
  --observer-root results/ca1m_port/ca1m_c3_native_b6_observer_fixed10_v1 \
  --diagnostics-root diagnostics/ca1m_port/ca1m_c3_native_b6_observer_fixed10_v1/native_b6 \
  --historical-prediction-root results/ca1m_port/c1_selective_boxer_fixed10_v2 \
  --historical-boxer-root diagnostics/ca1m_port/c1_selective_boxer_fixed10_v2 \
  --observer-boxer-root diagnostics/ca1m_port/ca1m_c3_native_b6_observer_fixed10_v1/boxer \
  --historical-log-root logs/ca1m_port/c01_paired_fixed10_v2/c1 \
  --observer-log-root logs/ca1m_port/ca1m_c3_native_b6_observer_fixed10_v1/inference \
  --output /tmp/ca1m_c3_native_b6_identity_audit_v2.json
```

Cache-replay fixed-ten FPS is a paired engineering measurement, not a claim
of original end-to-end real-time performance.  A later single-GPU, live-CuTR
run must independently demonstrate at least 10 FPS before making that claim.
