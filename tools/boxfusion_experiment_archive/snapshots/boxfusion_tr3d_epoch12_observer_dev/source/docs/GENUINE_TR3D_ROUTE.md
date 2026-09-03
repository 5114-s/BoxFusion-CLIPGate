# Genuine TR3D residual-proposal route

This directory is an isolated experiment. It reads the frozen B6 predictions
and ScanNet assets but never writes into either source directory:

- project: `/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev`
- frozen B6: `40.0434 / 33.5492 / 12.1613`
- frozen manifest: `manifests/frozen_b6_full100.json`
- pinned MMDetection3D: commit
  `fe25f7a51d36e3702f961e198894580d83c4387b`
- pinned official TR3D tree:
  `e7d4f3eaaeb39473babf52ef47af0d81fe72d6c8`

The route changes the proposal-recall ceiling. It does **not** yet append,
rescore or modify a BoxFusion prediction. No AP improvement is claimed until
a frozen observer audit passes the continuation gate.

## T0 and T1 are different experiments

### T0: official ScanNet-18 compatibility check

T0 uses the official 18-class TR3D config/checkpoint only to verify the pinned
runtime, CUDA operators, model construction and checkpoint integrity. It is a
closed-set ScanNet model. It is not the proposed class-agnostic detector, is
not an open-vocabulary result and is forbidden from writing residual caches.

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev

bash scripts/check_tr3d_environment.sh \
  "$PWD/.conda/boxfusion-tr3d" \
  --config "$PWD/third_party/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_scannet-3d-18class.py" \
  --build-model --require-cuda

python tools/verify_tr3d_checkpoint.py

python tools/verify_frozen_b6_manifest.py
```

The official 18-class checkpoint cannot directly initialize or masquerade as
the final single-class head. This checkout provides a separately manifested
foreground conversion for initialization only; see
`docs/TR3D_FOREGROUND_CHECKPOINT_INIT.md`. The immutable cache exporter still
rejects the original 18-class T0 output. The converted checkpoint satisfies
the one-class cache shape contract and can technically be used for an
initialization-only smoke, but it is forbidden as final T1/AP evidence.

For a fresh checkout, fetch and verify the pinned official checkpoint, then
create the deterministic foreground initialization:

```bash
bash scripts/fetch_tr3d_checkpoint.sh
bash scripts/convert_tr3d_foreground_init.sh
```

### T1: genuine class-agnostic TR3D

T1 retains the official TR3D sparse backbone, neck, regression, loss and NMS,
but replaces the semantic head/assignment with
`TR3DClassAgnosticHead`. Every ScanNet detection category is foreground
label `0`; both feature levels remain active. CLIP remains responsible for
open-vocabulary semantics after a proposal has survived the observer stages.

T1 must be trained on the official ScanNet **train** split only. The 1,201
train scenes are deterministically frozen as 1,001 gradient-train, 100
calibration and 100 audit scenes. All 312 official validation scenes are
forbidden for training and checkpoint selection.

## 1. Prepare and validate data

```bash
bash scripts/prepare_tr3d_scannet_data.sh

python tools/validate_tr3d_experiment.py \
  --mode full-train \
  --annotation data/tr3d_scannet/annotations/scannet_infos_train_foreground.pkl
```

The contract at `data/tr3d_scannet/DATASET_CONTRACT.json` records exact split
hashes, coordinate frames and the forbidden validation list. Points stay in
unaligned ScanNet world coordinates. The official pipeline applies
`GlobalAlignment` exactly once.

## 2. Train T1 on full scenes

Every launch requires a unique run tag. A non-empty work directory is refused
unless `BOXFUSION_TR3D_RESUME=1`; resume is allowed only in that same existing
work directory. The launcher runs the environment/model/data-sample checks
before starting distributed training.

```bash
bash scripts/train_tr3d_foreground_full.sh \
  0,1 tr3d_fg_full_seed0_v1
```

The default entry uses
`config/tr3d/tr3d_scannet_foreground_from_official_init.py`, verifies the
converted foreground checkpoint provenance/SHA, and loads it as initialization
without resuming the obsolete 18-class optimizer.

The default output is:

```text
work_dirs/tr3d/tr3d_fg_full_seed0_v1/
```

The launcher preserves the official global batch of 16 by default: one GPU
uses 16 samples and two GPUs use 8 samples per GPU. Override only when memory
requires it, and report the changed global batch:

```bash
BOXFUSION_TR3D_BATCH_PER_GPU=2 \
BOXFUSION_TR3D_NUM_WORKERS=2 \
bash scripts/train_tr3d_foreground_full.sh \
  0,1 tr3d_fg_full_seed0_bs2_v1
```

A compatible foreground initialization has been deterministically converted
from the pinned T0 checkpoint. The default launcher verifies and uses it:

```bash
python tools/verify_tr3d_foreground_checkpoint.py

bash scripts/train_tr3d_foreground_full.sh \
  0,1 tr3d_fg_full_seed0_init_v1
```

Only the two classifier tensors are folded; the other 260 tensors are
byte-exact. This artifact is still untrained and makes no AP claim. Do not
pass the unconverted official 18-class checkpoint here; its classifier shape
and semantic meaning differ from T1. Also use `load_from`, never `resume`,
because its retained optimizer state belongs to the 18-class source.

Training from scratch is permitted only as an explicit, separately tagged
ablation by overriding the config:

```bash
BOXFUSION_TR3D_CONFIG="$PWD/config/tr3d/tr3d_scannet_foreground.py" \
bash scripts/train_tr3d_foreground_full.sh \
  0,1 tr3d_fg_full_scratch_ablation_v1
```

Resume an interrupted run without deleting or replacing checkpoints:

```bash
BOXFUSION_TR3D_RESUME=1 \
bash scripts/train_tr3d_foreground_full.sh \
  0,1 tr3d_fg_full_seed0_v1
```

Do not select `epoch_*.pth` on official ScanNet val. Use only the frozen
calibration partition defined by the config.

### One-step training smoke

Before a long run, execute one real data/forward/loss/backward/optimizer step:

```bash
bash scripts/smoke_tr3d_train_step.sh \
  0 tr3d_train_step_smoke_v1
```

This uses batch 1, zero workers, one RepeatDataset pass and an
`IterBasedTrainLoop(max_iters=1)` with validation disabled. Its isolated output
is `work_dirs/tr3d_smoke/<run_tag>` and can never resume or share a full-run
work directory. A successful `iter_1.pth` proves the training path executes;
it is not a trained detector or accuracy result.

## 3. Optional trajectory-prefix fine-tuning

Export prefix points and visibility-filtered foreground annotations:

```bash
bash scripts/export_tr3d_prefix_train.sh
```

The exporter must create:

```text
data/tr3d_scannet/annotations/scannet_infos_prefix_train_foreground.pkl
```

Then start a new run from a frozen full-scene T1 checkpoint:

```bash
BOXFUSION_TR3D_BASE_CHECKPOINT="$PWD/work_dirs/tr3d/tr3d_fg_full_seed0_v1/epoch_12.pth" \
bash scripts/train_tr3d_foreground_prefix.sh \
  0,1 tr3d_fg_prefix_seed0_v1
```

Prefix supervision includes only boxes with observed depth-point support. It
never copies unseen full-scene boxes into an early trajectory prefix.

## 4. One-scene T1 smoke

This is the first command allowed to create a T1 residual cache. It requires a
trained one-class checkpoint and a new cache/log namespace:

```bash
BOXFUSION_TR3D_CHECKPOINT="$PWD/work_dirs/tr3d/tr3d_fg_full_seed0_v1/epoch_12.pth" \
bash scripts/run_tr3d_single_scene_smoke.sh \
  0 tr3d_t1_scene0568_smoke_v1
```

The script uses the real per-scene `axis_align_matrix`, exports corners back
to unaligned world coordinates, validates the immutable cache, and refuses
an existing output root. Its default static scene list is
`evaluation/data_util/meta_data/scannetv2_val_smoke1.txt`; override it with
`BOXFUSION_TR3D_SCENE_LIST` only when the replacement contains exactly one
scene.

The converted initialization has also been run through a ten-scene observer
as a pipeline/recall diagnostic. That run proves only that the class-agnostic
model, coordinate conversion, cache and audit chain execute. Because the
checkpoint had not undergone foreground training, its output is not a final
T1 result and cannot support an AP-improvement claim.

The diagnostic produced 2,197 candidates for 149 GT boxes. Its union-oracle
recall was `95.97%` at IoU 0.25 and `89.26%` at IoU 0.50, versus frozen-B6
oracle recall of `48.32%` and `29.53%`. However, the novel IoU-0.50 precision
upper bound was only `4.05%`, so the preregistered continuation gate failed.
This is evidence that TR3D changes the recall ceiling, while also showing why
foreground training and train-only score calibration are mandatory before
activation. The exact hashes and summary are frozen in
`manifests/tr3d_foreground_init_observer_smoke.json`.

## 5. Frozen 10-scene observer

After the smoke passes, run only the fixed ten-scene list:

```bash
BOXFUSION_TR3D_CHECKPOINT="$PWD/work_dirs/tr3d/tr3d_fg_full_seed0_v1/epoch_12.pth" \
bash scripts/run_tr3d_observer10.sh \
  0,1 tr3d_t1_observer10_v1
```

If a worker was interrupted, reuse only the exact checkpoint/config/cache:

```bash
BOXFUSION_TR3D_CHECKPOINT="$PWD/work_dirs/tr3d/tr3d_fg_full_seed0_v1/epoch_12.pth" \
BOXFUSION_TR3D_RESUME=1 \
BOXFUSION_TR3D_ATTEMPT_TAG=resume_001 \
bash scripts/run_tr3d_observer10.sh \
  0,1 tr3d_t1_observer10_v1
```

Resume validates every existing cache against the point file, checkpoint and
config hashes. It never overwrites a cache. Logs use a new attempt directory.

## 6. Audit union recall without changing B6

Use the exact checkpoint that produced the cache:

```bash
BOXFUSION_TR3D_CHECKPOINT="$PWD/work_dirs/tr3d/tr3d_fg_full_seed0_v1/epoch_12.pth" \
bash scripts/audit_tr3d_observer.sh \
  tr3d_t1_observer10_v1
```

The audit:

1. verifies all frozen B6 file/checkpoint/list hashes;
2. validates the exact cache artifact set;
3. aligns B6 and T1 boxes in the same coordinate frame;
4. reports score-ordered and maximum-cardinality union recall at
   IoU 0.15/0.25/0.50;
5. verifies frozen B6 hashes again after evaluation.

It has these hard invariants:

```text
observer_only=true
mutation_enabled=false
applied_count=0
class_agnostic=true
labels_3d[:]=0
```

Continue to later occupancy/grouping/multi-view confirmation only if the
frozen report passes its preregistered gate:

- union oracle `delta Recall@0.25 >= 0.08`;
- union oracle `delta Recall@0.50 >= 0.05`;
- at least five novel IoU-0.50 true positives in at least three scenes;
- novel AP50 precision upper bound at least 0.15.

If it fails, do not hide the result by tuning on those ten validation scenes.
First diagnose training convergence and proposal coverage on the disjoint
train/calibration partitions. A failed gate is evidence about the method, not
automatically a parameter problem.

## What this route does not claim

- T0 numbers are not T1 numbers.
- A successful cache export is not an AP gain.
- Oracle union recall is not deployable precision or AP.
- T1 is supervised on ScanNet train and must be described as a supervised
  proposal hybrid, even if CLIP supplies open-vocabulary labels.
- Training-free status is not preserved.
- Online speed is not assumed. Report the measured T1 runtime in cache
  diagnostics, total BoxFusion FPS and GPU memory before considering active
  deployment.
- A `+10` AP improvement is a target, not a guaranteed consequence of adding
  TR3D or of adding three modules.
