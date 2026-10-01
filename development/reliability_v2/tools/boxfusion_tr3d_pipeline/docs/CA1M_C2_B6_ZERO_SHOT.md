# CA-1M P2: frozen ScanNet-B6 zero-shot observer

This stage asks one narrow question: can the frozen ScanNet B6 quality head
rank CA-1M Selective-Boxer detections better without changing their geometry?

## Frozen parent

P1 is the existing fixed-ten CA-1M run:

```text
CA-1M C0 real score, score_thresh=0.4, gap=20
+ Selective Boxer G0 (center <= 0.10 m, volume ratio in [0.50, 2.00])
```

Its measured AP15/AP25/AP50 is `34.6810 / 29.3214 / 12.5369`.

## P2 observer contract

P2 adds live YOLOE prompt-free masks, true depth back-projection, object
memory, and masked CLIP evidence.  It runs the `quality_observer` profile, so
refit, neural refit, quality-score mutation, supplemental output, filtering,
and Soft-NMS are all disabled.  The runner saves a create-only snapshot
immediately before the online finalizer and requires the post-finalizer OBB
corners, detector scores, row count, and row order to be exactly identical in
that same process. Observer calls also run inside a restored Python/NumPy/
Torch RNG scope and are checked for direct mutation after every keyframe.

The older frozen P1 output is not used as the bitwise identity anchor:
independent GPU fusion replays can differ at floating-point level even when
the observer is a no-op.  It remains a historical metric and replay-drift
reference, with label/score/order and Selective-Boxer stable fields still
checked exactly.

The identity audit also checks the frozen P1/CuTR cache hashes, deterministic
Selective-Boxer fields, the 12-column quality schema, diagnostic-to-output row
mapping, and zero mutation counters.

After identity passes, an offline tool applies the frozen ScanNet-train B6
checkpoint only to observed rows:

```text
counterfactual_score = 0.40 * detector_score + 0.60 * B6_ranking_score
```

The score-only tree is marked `diagnostic_non_authoritative`; it is not an
active model result.  CA-1M OBBs are converted to enclosing world AABBs only
when forming the 12 B6 features, so this experiment is a cross-domain proxy,
not a native yaw-aware CA-1M quality head.

## Run

Preflight only (no output is created):

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
BOXFUSION_CA1M_C2_PREFLIGHT_ONLY=1 \
BOXFUSION_RUNTIME_TMP_ROOT=/tmp/ca1m_c2_preflight \
bash scripts/run_ca1m_c2_b6_zero_shot_observer.sh 0,1
```

Fixed-ten observer and counterfactual evaluation:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
bash scripts/run_ca1m_c2_b6_zero_shot_observer.sh 0,1
```

The run is create-only.  A partial namespace is never silently resumed; use a
new `BOXFUSION_CA1M_C2_RUN_TAG` after diagnosing a failed run.

## Decision rule

Do not enable B6 on CA-1M unless all of the following hold on the frozen
experiment:

- identity audit is 100% exact;
- observed-row coverage is at least 60%;
- AP15 change is at least -0.3 point;
- AP25 and AP50 each improve by at least +0.5 point;
- a later no-cache, single-GPU end-to-end run remains at least 10 FPS and is
  no more than 10% slower than its paired P1 runtime.

If B6 ranking is consistently worse and its features are strongly outside the
ScanNet training distribution, the failure is a domain/geometry-contract
problem rather than a detector-score threshold problem.  In that case the
ScanNet B6 checkpoint must not be made active on CA-1M.
