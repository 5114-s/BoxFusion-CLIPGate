# BoxerNet-only lifting ablation

## Question

This branch tests one narrow question:

> When the RGB frame, real ScanNet depth, camera calibration, CuTR 2D
> proposals, detector scores, proposal order, and the complete downstream
> BoxFusion pipeline are fixed, does official BoxerNet produce better 3D
> lifting geometry than CuTR?

It does **not** add B6, CLIP gating, Top-K view fusion, YOLOE/SAM proposals,
TR3D, or any learned score calibration.  Therefore an AP change can be
attributed to the lifting geometry.

The primary active profile applies Boxer after the unchanged CuTR
score/UV/floor filters. This is intentionally a controlled geometry-only
ablation: every 2D row entering association is frozen. R0 records those rows
into an immutable per-keyframe cache and immediately consumes the
deserialized copy. Formal X0, X1 and X2 replay the same cache. X0/X1 both
execute the identical non-mutating Boxer observer workload, so CUDA path
differences cannot masquerade as geometry changes. Thus the comparison
measures lifting quality without CuTR's cross-process GPU nondeterminism.
CuTR is still the source of the 2D proposals, so this is not a replacement for
proposal generation.

If the controlled experiment is positive, a second, separately reported
experiment may set `apply_stage: pre_filter`.  That is the full lifting
replacement in which Boxer geometry is also allowed to affect the unchanged
UV/floor filters.  Results from the two stages must not be mixed.

## Frozen components

- official BoxFusion source base: commit `9f9cda0`
- official Boxer source: commit
  `1f86542dc342a4b1d474c87c97c5d1d6566d9148`
- Boxer checkpoint:
  `boxernet_hw960in2x6d768-c88128f8.ckpt`
- Boxer checkpoint SHA256:
  `d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f`
- DINOv3 checkpoint SHA256:
  `4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea`
- ScanNet keyframe gap: 25
- real detector score threshold: 0.5
- random/evaluation seed: 0
- original 3D NMS, 2D correspondence, fusion, CLIP, and evaluation settings

Boxer is licensed CC-BY-NC-4.0.  This route is for non-commercial research and
must retain the required attribution.

## Profiles

| Profile | Behavior | Required result |
|---|---|---|
| R0 `x0_cutr` | cache source: records filtered CuTR rows | not the denominator |
| X0 `x0_replay` | replays CuTR geometry and runs non-mutating Boxer observer | reference |
| X1 `x1_observer` | repeats the exact X0 observer workload | identical AP and bounded fusion noise |
| X2 `x2_active` | replays X0 rows; replaces only 3D box geometry | measure controlled AP delta |
| F1 `f1_pre_observer` | pre-filter observer; cannot mutate predictions | byte-identical to X0 |
| F2 `f2_pre_active` | full pre-filter lifting replacement | secondary AP delta |

For every Boxer call, diagnostics preserve hashes of RGB, depth, both
intrinsics, camera pose, 2D boxes, scores, all protected instance fields,
CuTR geometry, Boxer geometry, source commit, and checkpoint.

The contract is fail-closed: a missing row, count mismatch, stale prediction,
NaN/Inf, non-positive dimension, invalid rotation, missing diagnostic, or
checkpoint/source mismatch aborts the scene.  No silent CuTR fallback is
allowed in X2.  Boxer uncertainty is recorded only and cannot change scores or
filtering.

The proposal cache also stores the complete CuTR field schema/order, tensor
dtype/shape/hash, 3D box DOF, primary-versus-retry source, empty events,
RGB-D/calibration hashes, RNG state, dataset length/gap, and the frozen X0
prediction hash. Replay must consume every recorded keyframe in order. A
partial or stale cache fails closed and is never overwritten.

## Run

Prepare the isolated read-only links and verified official weights once:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev
bash scripts/prepare_boxer_lifting_assets.sh
```

Do not launch while another experiment occupies both GPUs.  After the GPUs are
idle, run one-scene smoke:

```bash
bash scripts/run_scannet_boxer_smoke.sh 0 scene0568_00
```

Then run the paired fixed-10 ablation:

```bash
bash scripts/run_scannet_boxer_fixed10.sh 0,1
```

The decision report is:

```text
reports/boxer_lifting/fixed10_summary.json
```

Only when `recommend_full100` is `true`:

```bash
bash scripts/run_scannet_boxer_full100.sh 0,1
```

The same positive gate also permits the secondary, true pre-filter replacement:

```bash
bash scripts/run_scannet_boxer_full_lifting_fixed10.sh 0,1
```

F2 keeps the UV/floor algorithms and thresholds unchanged, but they now
consume Boxer geometry.  Because this can naturally change whether the
original first-frame low-threshold retry is reached, diagnostics use the key
`(frame_id, attempt_id)` and report retry-only schedule differences.  Primary
CuTR 2D rows must still match exactly.

The 100-scene runner refuses to start when the fixed-10 gate failed.  A forced
run requires an explicit, documented override:

```bash
BOXFUSION_BOXER_FORCE_FULL100=1 \
bash scripts/run_scannet_boxer_full100.sh 0,1
```

## Interpretation

The historical true-score baseline `32.53 / 27.11 / 9.46` is context only.
Source and deterministic state changed after that artifact was produced, so
the paired X0 generated by this branch is the only valid denominator.

X1 must preserve prediction count/order/classes/scores exactly, have identical
AP at every threshold, and keep final box-coordinate drift at or below the
frozen `1e-4 m` tolerance. The tolerance exists only because the released
BoxFusion PyCUDA kernel sums per-view values with `atomicAdd_system`; repeated
identical workloads showed a maximum `3.99e-5 m` drift. All cached inputs and
adapter-boundary hashes remain byte-exact. A larger drift invalidates the test
and X2 must not be interpreted.

X1/X2 replay timing is not an online FPS result because those profiles skip
the CuTR forward call. If the AP gate passes, runtime must be measured
separately with live CuTR 2D proposals plus Boxer lifting.

The fixed-10 gate recommends a 100-scene run only when the contract passes,
X1 is identical, AP25 and AP50 each improve by at least 0.5 point, AP15 does
not decline by more than 0.5 point, and recall improves by at least 2 points
at IoU 0.25 or 1 point at IoU 0.50.  For a final claim that Boxer lifting is
useful, the locked 100-scene run should improve AP25 and AP50 by at least 1
point each while AP15 does not decline by more than 0.5 point.  Smaller
differences should be reported as neutral rather than tuned on validation
scenes.
