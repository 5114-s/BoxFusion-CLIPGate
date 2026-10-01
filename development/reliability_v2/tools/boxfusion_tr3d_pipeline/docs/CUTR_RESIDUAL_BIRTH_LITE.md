# CuTR residual-birth-lite (S0 shadow)

This step opens a training-free, causal missed-object branch without changing
the released BoxFusion result. It observes low-confidence rows from the same
live CuTR forward pass and reports terminal candidates; it does **not** append,
delete, rescore, or refine any native box.

## Frozen input contract

- On a normal CuTR attempt the residual interval is
  `[0.10, detection.score_thresh)`. In the supplied ScanNet config this is
  `[0.10, 0.50)`; `0.50` remains in the native path.
- `cutr_residual_birth_lite.score_ceiling` must equal
  `detection.score_thresh` exactly. A mismatch fails before inference.
- Only raw outputs from a live CuTR call are eligible. Proposal-cache record
  and replay are rejected while this observer is enabled because the existing
  cache stores already-filtered native proposals and cannot reconstruct rows
  below the native threshold.
- Raw row indices are retained in a separate audit vector. Residual rows are
  cloned before native lifting or filtering, then mirror the filters of that
  exact native attempt and receive an independent camera-to-world transform.
  The primary attempt mirrors UV-bound and floor filters; the released retry
  path has no floor filter, so its residual clone mirrors UV-bound only.
- Residual rows never enter the lifting adapter, CLIP, Moon-QIM, PUF,
  MV3DIS, BoxManager, BoxFusion, native post-processing, or export.
- This exact filter-parity statement is scoped to the supplied baseline
  `lifting.backend=cutr`/no-adapter configuration. Combining the observer with
  Selective Boxer would require a separately specified side-geometry contract
  and is intentionally not authorized here.
- On a frame-0 native retry, primary-attempt residual rows are discarded and
  only retry-attempt rows below the retry's *actual* native cutoff are
  observed. With this config that retry-only interval is `[0.10, 0.125)`;
  therefore no side row is simultaneously accepted as native. If the retry
  cutoff is at or below `0.10`, its residual set is empty. Every true CuTR
  keyframe is committed exactly once, including a keyframe with zero native or
  residual rows. The terminal stale-frame branch is never treated as new
  evidence.

## Frozen causal observer

The implementation is NumPy-only after proposal extraction and has no learned
weights, labels, ground truth, online optimization, or future-frame access.
Association uses the preceding observation only: AABB IoU at least `0.10` and
center displacement at most `0.50 m`. A track needs three distinct keyframe
observations and expires after ten missed keyframes. Expired unconfirmed tracks
are reclaimed; expired confirmed evidence is kept in a separately bounded
archive. Per-frame deduplication, active-track/observation caps, vectorized
geometry, and deterministic tie-breaking bound online work.

At terminal time the observer applies fixed geometry-stability, minimum-size,
native-novelty, residual self-NMS, and output-cap gates. `close()` receives
copies of the native boxes and scores *after* ScanNet native post-processing.
Its candidates are counterfactual audit records only. RNG state and native
proposal/output arrays are checked around observer calls.

## Running the shadow

Use `config/scannet_cutr_residual_shadow.yaml` with the normal live ScanNet
runner. The log contains one machine-readable line:

```text
CuTR-residual-birth-lite shadow JSON | {...}
```

`observer_only=true`, `active_authorized=false`,
`native_mutation_applied=false`, and `native_export_appended=false` are the
safety conditions for this stage.

## Accuracy status

The first real paired run used `scene0598_01`, live CuTR, seed 0, and one GPU.
The native control ran at 32.53 FPS and the shadow observer at 32.50 FPS
(99.91% retention). The NumPy core p95 was 1.21 ms per CuTR keyframe and the
one-time terminal close was 10.48 ms; core timing excludes proposal cloning,
filtering, world transform, copies, and identity guards, so the end-to-end FPS
pair is the authoritative real-time measurement.

The observer received 952 low-score rows, confirmed 90 tracks, and emitted six
counterfactual candidates with a complete audit. A create-only artifact kept
all 16 native rows as an exact prefix and appended those six at scores below
the native floor. Official deterministic single-scene evaluation gave:

| output | AP15 | AP25 | AP50 | P@15/P@25 | R@15/R@25 |
|---|---:|---:|---:|---:|---:|
| native | 0.277389 | 0.277389 | 0.030303 | 0.312500 | 0.454545 |
| +6 residual | 0.277389 | 0.277389 | 0.030303 | 0.227273 | 0.454545 |

Thus none of the six candidates added a true positive at IoU 0.15, 0.25, or
0.50. AP and recall stayed unchanged while precision fell. The maximum GT IoU
of any candidate was only 0.045899. This rule remains observer-only and must
not be expanded to fixed-10 or activated as an append policy. The next shadow
iteration should add training-free cross-view evidence (viewpoint diversity
plus CuTR descriptor/depth consistency) before any candidate reaches the
terminal append set. The paired report is
`reports/cutr_residual_birth/scene0598_s0_v1.json`.
