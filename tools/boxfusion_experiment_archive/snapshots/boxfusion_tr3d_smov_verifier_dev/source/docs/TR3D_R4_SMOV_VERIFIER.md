# R4 SMOV3D-inspired verifier

## Scope

R4 is a harm-verifier for the already selected terminal R3 replacements.  It
does not propose boxes, change scores, change labels, call CLIP, or change the
prediction order/count.  The implementation is isolated from the frozen R3
route and uses create-only evidence sidecars.

The name is deliberately **SMOV3D-inspired**.  The local workspace does not
contain the official SMOV3D implementation, and the support/occluded/
free-space ray classifier is our concrete implementation of its multi-view
geometric-consistency principle.

## Implemented ablations

1. `R4-D`: candidate and original G0 anchor are evaluated on a common causal
   Top-K using real metric depth.
2. `R4-FS`: support, occluded, free-space and invalid ray evidence is retained
   per view and in aggregate.
3. `R4-F`: support points are projected to RGB and pooled from the official
   Boxer DINOv3-S/16+ dense map.  The standalone exporter is diagnostic only;
   a future live path must reuse Selective Boxer's existing `dino0` and is not
   allowed to execute a second backbone.
4. A GT-only offline counterfactual audit measures each replacement by
   restoring its G0 geometry from the joint raw-R3 output.  GT is never
   imported by either inference observer.

All exporters verify the active prediction-tree hash before and after work.

## Frozen fixed-10 result

The fixed ten-scene run contained 59 selected replacements: 24 gain, 2 harm,
and 33 neutral under rank-aware leave-one-out evaluation.

| configuration | AP15 | AP25 | AP50 | veto/harm/gain |
|---|---:|---:|---:|---:|
| same-run G0 | 44.6302 | 40.8154 | 17.1297 | - |
| raw terminal R3 | 45.2499 | 44.0374 | 31.4913 | - |
| R4-D support | 44.6302 | 42.9954 | 22.6807 | 21 / 1 / 11 |
| R4-FS support+free | 44.6302 | 44.0374 | 26.3991 | 13 / 1 / 6 |
| R4-F strict DINO dominance | 45.2499 | 44.0374 | 31.4913 | 0 / 0 / 0 |
| perfect harm-veto upper bound | 45.2499 | 44.0374 | 32.5901 | 2 / 2 / 0 |

The fixed unsupervised rules fail the pre-registered activation gate.  Depth
support/free-space removes too many beneficial tighter candidates, while the
strict DINO dominance rule never fires.  Do not materialize an active R4 from
these validation results and do not tune a validation threshold to rescue it.

## Reproduction

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_smov_verifier_dev

BOXFUSION_R4_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt" \
BOXFUSION_R4_ARTIFACT_BASE=/extra/ZhaoX/codex_artifacts/boxfusion_r4_smov_fixed10 \
bash scripts/run_tr3d_r4_depth_observer.sh r4d_fixed10_new_v1

BOXFUSION_R4_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt" \
BOXFUSION_R4_ARTIFACT_BASE=/extra/ZhaoX/codex_artifacts/boxfusion_r4_smov_fixed10 \
bash scripts/run_tr3d_r4_feature_observer.sh \
  0 r4d_fixed10_new_v1 r4f_fixed10_new_v1

BOXFUSION_R4_ARTIFACT_BASE=/extra/ZhaoX/codex_artifacts/boxfusion_r4_smov_fixed10 \
bash scripts/audit_tr3d_r4_fixed10.sh \
  r4d_fixed10_new_v1 r4f_fixed10_new_v1 r4_counterfactual_fixed10_new_v1
```

Use new tags because every cache/report is immutable.
