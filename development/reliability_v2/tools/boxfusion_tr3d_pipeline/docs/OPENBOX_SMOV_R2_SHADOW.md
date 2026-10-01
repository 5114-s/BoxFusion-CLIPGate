# OpenBox-SMOV R2 shadow geometry

OpenBox-SMOV R2 is a training-free, causal geometry observer for the native
CuTR + BoxFusion stream.  It retains a bounded set of RGB-D observations for
each stable native track and emits terminal counterfactual boxes.  It does not
modify native association, fused geometry, detector scores, row order,
categories, or CLIP semantics.

## Two-phase online contract

On each real CuTR keyframe (`count % gap == 0`), `demo.py` performs exactly two
R2 operations:

1. After CuTR boxes are transformed into metric world coordinates, and before
   any native association, it defensively copies proposal IDs, raw image-space
   `xyxy` boxes, depth in metres, the current depth intrinsics, and the current
   camera-to-world pose.  Empty current proposal sets create an explicit
   abstention; the legacy terminal replay is never treated as a new view.
2. After native association and BoxFusion finish, it commits the frozen batch
   with the native merge trace and final fusion groups.  R2 owns an independent
   causal stable-ID registry, so it also works when Moon-QIM-lite is disabled.

Only one observation from a `(stable track, frame)` pair is retained.  Memory
is bounded to five views and 1,024 points per track, with at most 512 points
from any one view.  Invalid sensors, ambiguous lineage, inadequate parallax,
too few views/points, budget exhaustion, and unsafe geometry all abstain.

## Counterfactual-only terminal output

Native ScanNet extent filtering runs first.  The same `valid_mask` is applied
to R2 stable IDs before `finalize_shadow`.  The observer evaluates bounded
same-yaw quantile and PCA-yaw quantile hypotheses using held-out-view evidence.
For each yaw it additionally evaluates the fixed recipe lattice
`{base, face_x, face_y, face_xy}`.  A face recipe is available only when every
LOO fold independently finds the same extension direction and the final
all-view fit agrees.  The selected counterfactual is written only to a
create-only `.npz` sidecar under `openbox_smov_r2.diagnostics.root`.

## OpenBox face-visibility adaptation

This v2 observer borrows the ray/face-normal direction test from OpenBox's
visibility-based box extension.  On the four local XY faces, each retained
view computes the dot product between the outward normal and the normalized
face-to-camera ray.  The existing sparse depth component acts as a cheap
surface proxy: a face is strong at dot `>=0.25` with at least eight band
points, and weak at dot `>=0.05` with at least four.  Evidence is an any-view
union; point counts are never pooled across views to manufacture a strong
face.

An axis is extendable only when one face is strong and its opposite has no
weak evidence.  The observed face remains fixed while the unseen bound moves
outward by `clip(0.25 * extent, 0.05 m, 0.30 m)`.  Z extent and yaw are not
changed by this operation.  Up to four recipes per yaw remain subject to the
same native-relative center/extent safety checks, multi-view crop IoU, depth
support, and free-space veto as the base quantile candidates.

This is an **OpenBox-inspired geometry-only adaptation**, not a reproduction
of the complete paper pipeline.  The original method uses offline SDF meshes,
physical-state routing, dynamic trajectories, and category-size statistics.
Those parts are intentionally absent because they do not fit this observer's
causal, semantic-isolated, real-time contract.  The retained correspondence
is the XY face visibility rule plus multi-view 2D selection described in the
[OpenBox paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/3206d39c0680c8e953ff136ed7f84e0f-Paper-Conference.pdf).

`boxes_3d` is never assigned from the R2 result.  Prepare receives independent
CPU copies and checks those source copies exactly; commit receives fresh
group/event values rather than mutable native BoxManager objects.  Terminal
finalization retains exact `numpy.array_equal` checks for native geometry,
scores, stable IDs, and row order.  The normal
`<scene>_boxes.pkl` remains the unmodified BoxFusion prediction, so shadow AP
is intentionally identical to the control.  A precision gain can only be
claimed after a separate, preregistered active experiment.

## Isolation and runtime

The frozen config is `config/scannet_openbox_smov_r2_shadow.yaml`, copied from
`config/scannet_eval.yaml` plus the single R2 block.  Proposal-cache
record/replay and `online_refinement` are rejected while R2 is enabled; no
terminal R3 or learned quality/refinement component is enabled.  CLIP follows
the original code path unchanged and is not visible to R2.

Example paired run (use a fresh output and diagnostics root for every run):

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline
python demo.py scannet \
  --model-path ../../models/cutr_rgbd.pth \
  --config ./config/scannet_openbox_smov_r2_shadow.yaml \
  --device cuda --seed 0 --seq scene0277_00
```

Compare against the same command with `./config/scannet_eval.yaml`.  Before
considering activation, require exact native prediction identity, zero stale
terminal observations, no duplicate same-frame views, a same-device FPS ratio
of at least `0.95`, and held-out evidence that the proposed replacements help
AP50 rather than merely refitting their own support points.

The paired artifacts can be checked without ground truth or materializing an
active prediction:

```bash
python tools/audit_openbox_smov_r2_shadow.py \
  --contract visibility-v2 \
  --scene-list evaluation/data_util/meta_data/scannetv2_val_smoke_scene0277_00.txt \
  --prediction-root results/openbox_smov_r2/scene0277_visibility_v2 \
  --diagnostics-root diagnostics/openbox_smov_r2/scene0277_visibility_v2 \
  --log-root logs/openbox_smov_r2/scene0277_visibility_v2 \
  --anchor-root results/openbox_smov_r2/scene0277_control_v1 \
  --control-log-root logs/openbox_smov_r2/scene0277_control_v1 \
  --report results/openbox_smov_r2/scene0277_visibility_v2/audit.json
```

The sidecar schema is `boxfusion.openbox_smov_r2_shadow.v2`.  The five partial
full100 artifacts previously produced under `full100_shadow_seed0_v1` use the
old v1 candidate rules and must not be resumed into a v2 run.  Use fresh
native prediction, diagnostics, and log roots.  The same audit executable can
still inspect those old artifacts explicitly with `--contract r2-v1`; one
audit invocation never accepts a mixed v1/v2 scene list.

The earlier 2026-08-21 `scene0277_00` **v1** smoke run passed its audit: native prediction
bytes were identical to the paired control, shadow/control throughput was
33.63/33.95 FPS (`0.9906` ratio), wrapper p95 was 7.56 ms per keyframe, and
all eight terminal rows safely abstained because the retained views did not
meet the frozen mutual pose-diversity rule.  Consequently this smoke run
validates isolation and runtime, not an accuracy gain.  It does not validate
the new v2 face candidates; v2 requires a fresh paired smoke run.

The final v2 `scene0412_00` regression used GPU 1 and a fresh artifact root.
It ran at 24.18 FPS versus 24.02 FPS for the adjacent control (`1.0067` ratio),
with core/wrapper p95 of 4.68/6.31 ms.  The immutable
`visibility-v2` audit passed prediction-to-sidecar identity, frozen caps,
receipt semantics, LOO dominance, and paired realtime checks.  One
`native_yaw_quantile+face_y` candidate extended the unseen local +Y bound by
0.181 m, but projection IoU, depth support, and free-space evidence did not
jointly dominate the native box, so it was correctly rejected.  A separate
base quantile candidate remained a would-replace counterfactual; neither box
was applied to the native output.  This validates that the OpenBox-inspired
branch executes and abstains safely, but it is still not evidence of an AP
gain.

## Full100 counterfactual AP result (2026-08-22)

The visibility-v2 observer was subsequently rerun on all 100 unique scenes in
`evaluation/data_util/meta_data/scannetv2_val.txt` with seed 0 and frozen
core/demo/config/runner hashes.  The fresh run produced exactly 100 native
pickles and 100 v2 sidecars.  A GT-free structural audit found 1,793 terminal
rows and ten `would_replace` rows.  The separate create-only materializer
changed only those ten geometries; labels, score bits, row count, and row order
were preserved.

The unchanged ScanNet evaluator was then run as a paired experiment.  Both
commands used the same 100-scene order, GT, seed, and evaluator hash; only the
prediction and dump roots differed.

| output | AP15 | AP25 | AP50 |
|---|---:|---:|---:|
| same-run native | 32.3579 | 27.3087 | 9.5001 |
| R2 counterfactual | 32.3734 | 27.2161 | 9.4442 |
| delta (percentage points) | +0.0155 | -0.0926 | -0.0559 |

This result rejects active materialization of the current R2 policy.  The ten
replacements comprised six quantile-base hypotheses and four visibility-face
hypotheses.  GT-only post-evaluation attribution located both threshold losses
in the base branch: one box fell from IoU 0.3076 to 0.2257 at AP25 and another
from 0.5944 to 0.4751 at AP50.  None of the four face extensions changed a
threshold-level TP assignment.  Consequently the OpenBox-inspired face idea
is not shown to improve AP by this run, while the combined R2 route is mildly
harmful at AP25/AP50 and remains shadow-only.

The full100 run also missed the preregistered absolute 20 FPS-per-scene floor
(minimum scene FPS 8.90).  The relaxed-latency audit cited below is an artifact
identity/receipt audit only, not a realtime pass; the earlier paired smoke
tests remain the available relative-overhead evidence.

Immutable evidence:

- `reports/openbox_smov_r2/full100_visibility_v2_seed0_r1/paired_ap.json`
- `reports/openbox_smov_r2/full100_visibility_v2_seed0_r1/materialization.json`
- `reports/openbox_smov_r2/full100_visibility_v2_seed0_r1/shadow_integrity_relaxed_runtime.json`
- `logs/openbox_smov_r2/full100_visibility_v2_seed0_r1_paired_ap/{native,counterfactual}.log`

The offline tools are:

- `tools/materialize_openbox_smov_r2_counterfactual.py`
- `tools/evaluate_openbox_smov_r2_counterfactual.py`

They are deliberately separate: materialization has no GT/evaluator input,
and GT is opened only by the paired AP driver after the geometry-only output
tree has been sealed.
