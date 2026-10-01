# B6 + SGCDet-inspired local sparse refiner

## Scope and scientific claim

Official SGCDet is **not** an object-local BoxFusion refiner.  It is a complete
supervised multi-view 3D detector: an image backbone and depth head build a
scene-level feature volume, a coarse-to-fine sparse volume module refines
likely occupied voxels, and a 3D detection head predicts boxes.  Replacing a
BoxFusion function with the official detector would therefore be a different
pipeline, not a clean one-module ablation.

This isolated route adapts one idea only: SGCDet's coarse-to-fine occupancy and
Top-K sparse refinement.  B6 detections and all BoxFusion semantics remain the
anchor.  Each B6 box defines a canonical object-local voxel grid.  Multi-view
real-depth evidence accumulated for that box is voxelised; coarse occupancy
selects Top-K cells; a lightweight local sparse encoder predicts a bounded
geometric residual and its quality/uncertainty.  The active profile may change
box geometry only.  It must not change detection count, instance order, B6
score, or CLIP label.

The experiment lives entirely in
`boxfusion_b6_sgcdet_local_refiner_dev`.  It does not edit or write results into
the U2 route.

## Exact upstream mapping

The audited upstream checkout is fixed at commit
`eb4ba52a711ab30302569ce7329aca9be28aa39d`; see
`third_party/SGCDet_PROVENANCE.md`.

| Official SGCDet component | Upstream evidence | Local adaptation |
|---|---|---|
| Three-resolution volume | `configs/SGCDet_ScanNet.py`: voxel sizes `(0.64,0.64,0.8) -> (0.32,0.32,0.4) -> (0.16,0.16,0.2)` | Normalised box-local coarse/fine grids; resolution is independent of absolute object size. |
| Occupancy prediction | `AdaptiveSparseHead.occ_pred_heads`, trained with geometric occupancy | A class-agnostic occupancy score for local RGB-D voxels. |
| Hard Top-K selection | `AdaptiveSparseHead.topk_wo_grad` and ScanNet `topk_list=[800,6400]` | Top-K local voxels are selected at each refinement level; K is scaled to the much smaller object-local grid. |
| Coarse-to-fine residual construction | `upsampled_volume + base_heads[i](..., proposal=indices_current)` | Fine local features are evaluated only around selected occupied cells and combined with coarse context. |
| Geometry/context aggregation | `DenseHead` and `DeformCrossAttention_DFA3D` | Existing BoxFusion multi-view real-depth observations provide geometry and view statistics; no second image backbone is introduced. |
| 3D neck and detector head | `FastIndoorImVoxelNeck` + `ScanNetImVoxelHeadV2` | Replaced by a small class-agnostic residual head for `delta_center`, `delta_log_extent`, candidate IoU and uncertainty. |

The official Top-K values and scene voxel sizes are not copied numerically:
they describe a room-scale grid and are inappropriate for an object crop.

## Why the official CUDA/MMCV stack is not imported

The pinned SGCDet install recipe fixes PyTorch 1.10.1, CUDA 11.3,
`mmcv-full==1.5.3`, `mmdet==2.25.1`, and a vendored MMDetection3D.  Its DFA3D
attention path loads compiled `_ext` CUDA operators.  The BoxFusion B6 runtime
uses a newer PyTorch/CUDA stack.  Loading both stacks in one process risks an
ABI mismatch, and running the complete SGCDet backbone would invalidate the
local-module and online-runtime ablations.  Consequently:

- `third_party/SGCDet` is never added to `PYTHONPATH`;
- no SGCDet checkpoint or compiled operator is required;
- the local head uses the route's native PyTorch only;
- the official code is consulted for architecture/provenance, not imported.

## Method status

This route is a **ScanNet-train supervised geometric hybrid**.  Training uses
only official ScanNet train scene boxes and never validation scenes.  It is not
correct to describe the active refiner as target-dataset training-free or
zero-shot on ScanNet.

The head is class-agnostic: it neither consumes nor predicts an 18-class label.
CLIP/open-vocabulary text scoring remains unchanged, so the semantic interface
is still open-vocabulary.  This distinction must be reported explicitly:
open-vocabulary semantics do not make the new geometry head training-free.

## Leakage and output contracts

1. The collection and training scripts reject a path whose name contains
   `val` and reject any scene ID shared with `scannetv2_val.txt`.
2. Train diagnostics, datasets, checkpoints, predictions, logs, and reports
   are rooted in this isolated directory unless explicitly overridden.
3. S0 is frozen B6.
   Its protocol is fixed to B6 quality/detector blend `0.40` and ScanNet
   minimum output extent `0.40 m`, matching the recorded
   `40.0434 / 33.5492 / 12.1613` full100 anchor.
   The source prediction hashes, scene list, checkpoint hash and reference
   result path are recorded in `manifests/frozen_b6_full100.json`.
4. S1 observer collects sparse features/targets and must be an exact no-op
   inside its own run.
5. S2 loads an identity checkpoint and must also be an exact no-op inside its
   own run.
6. S3 is the only profile allowed to alter box corners.  Count, order, B6 score,
   and semantic label remain fixed.
7. A 100-scene evaluation is rejected unless its scene list and a fresh run
   tag are explicitly set with `BOXFUSION_SGCDET_SCENE_LIST` and
   `BOXFUSION_SGCDET_RUN_TAG`.  Do fixed10 first.

Do not compare pickle bytes from independent GPU reruns.  BoxFusion's custom
PyCUDA fusion uses parallel floating-point atomics, so identical seeds do not
guarantee bitwise-identical boxes across processes.  The audit therefore
checks exact pre/post/export identity against each run's own diagnostics.
Cross-run count, matching, numeric, and ranking differences are report-only
warnings because proposal-score and NMS boundary flips can also change the
output set.  AP equality remains a secondary control, and small cross-run AP
changes must not be interpreted as observer gain.

## Protocol

The scripts do nothing merely by being installed.  A GPU job starts only when
the user explicitly invokes a collection or evaluation command.

### 1. Collect train-only observer diagnostics

Train-only collection deliberately uses the already audited
`b5v2_memory_observer` provenance contract because the strict AP50 dataset
join recognizes that exact K=5/P=128 observer. It is still output-preserving;
`sgcdet_sparse_observer` is reserved for the paired S1 validation control.

Prepare the official ScanNet train RGB-D frames first.  Then run:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_sgcdet_local_refiner_dev

BOXFUSION_SGCDET_TRAIN_SCENES="$PWD/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt" \
BOXFUSION_SCANNET_FRAMES_ROOT="$PWD/data/scannet_train" \
bash scripts/collect_scannet_sgcdet_sparse_train.sh 0,1
```

The default B6 quality checkpoint is
`models/scannet_b6_iou_mlp.npz`.  If it is intentionally stored elsewhere,
provide a read-only override:

```bash
BOXFUSION_QUALITY_CHECKPOINT=/absolute/read-only/path/scannet_b6_iou_mlp.npz \
  bash scripts/collect_scannet_sgcdet_sparse_train.sh 0,1
```

The route also pins its copied YOLOE provider weight by default. An explicit
replacement must use `BOXFUSION_SGCDET_YOLOE_CHECKPOINT`, preventing stale U2
shell variables from silently changing this ablation.

### 2. Build the train dataset and train on CPU

```bash
bash scripts/train_scannet_sgcdet_sparse_refiner.sh
```

This produces `models/scannet_sgcdet_sparse_refiner.pt` and the separate
`models/scannet_sgcdet_sparse_refiner_identity.pt` control checkpoint.
Training is CPU-only by default and does not occupy U2 GPUs.

### 3. Fixed10 incremental ablation

Run one stage at a time:

```bash
bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s0 0,1
bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s1 0,1
bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s2 0,1
bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s3 0,1
```

After S0--S2 finish, enforce the same-run identity contract before
interpreting S3:

```bash
bash scripts/audit_scannet_sgcdet_sparse_identity.sh
```

The stage mapping is:

| Stage | Profile | Sparse checkpoint | Permitted mutation |
|---|---|---|---|
| S0 | `quality_only` | none | frozen B6 only |
| S1 | `sgcdet_sparse_observer` | none | none; collect diagnostics |
| S2 | `sgcdet_sparse_identity` | identity | none |
| S3 | `sgcdet_sparse_active` | learned | box geometry only, behind its quality gate |

To override the S2/S3 checkpoint use
`BOXFUSION_SGCDET_SPARSE_CHECKPOINT=/absolute/path/checkpoint.pt`.

### 4. Full100 only after controls pass

There is intentionally no implicit full100 default.  Set the scene list and a
new run tag explicitly:

```bash
BOXFUSION_SGCDET_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_SGCDET_RUN_TAG=sgcdet_sparse_s3_active_full100_v1 \
bash scripts/run_scannet_b6_sgcdet_sparse_refiner.sh s3 0,1
```

Do not reuse a fixed10 output directory.  Full100 is justified only if the S1
and S2 same-run no-mutation contracts pass and S3 improves a paired identity
counterfactual from the same S3 run (plus held-out train-only development
scenes) without materially breaking runtime. Independent-run structure is a
diagnostic, not a gate.

## What a positive result would mean

This module directly targets localisation/AP50, not missing-proposal recall.
It can outperform B6 only when B6 already detects an object and the local
multi-view depth contains cleaner boundary evidence than its current box.  A
negative result is evidence that local evidence or residual gating is
insufficient; it is not a reason to append more modules without an oracle
analysis.  No precision gain is guaranteed before the fixed10 and full100
measurements.
