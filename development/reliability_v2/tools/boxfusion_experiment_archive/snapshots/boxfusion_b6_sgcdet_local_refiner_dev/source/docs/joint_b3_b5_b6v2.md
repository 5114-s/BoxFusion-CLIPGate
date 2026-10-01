# Joint B3 → B5 + B6-v2 Local Head

This branch is an isolated research implementation. It does not modify the
live `/data/ZhaoX/BoxFusion` repository and none of the commands below is
started automatically.

## What is joint

The old pipeline applied three largely independent operations:

1. B3 concatenated selected Mask-RGBD views into one geometry cloud.
2. B5 predicted a local centre/size residual from that flattened cloud.
3. B6 ranked the exported box with a separate 12-feature NumPy MLP.

The joint head retains the view axis and performs one shared forward pass:

```text
K=5 reliable Mask-RGBD views
            │
            ├── per-view local points [5, 128, 3]
            ├── per-view reliability [5, 9]
            └── legacy global quality features [12]
                              │
                    shared point/view encoder
                       ┌──────┴──────┐
                       │             │
                local box residual   original/candidate
                + P(improvement)     IoU/Q15/Q25/Q50
                       │             │
                       └──────┬──────┘
                     geometry-safe gate
                              │
               score branch matching exported box
```

The nine per-view attributes are view quality, mask confidence, valid-depth
ratio, projection IoU, sampled-point ratio, camera-valid flag, and the three
components of the local camera-to-object direction.

For this first speed-preserving ablation, masked CLIP appearance extraction is
disabled in both collection and active inference. The legacy 12-dimensional
feature named `appearance_consistency` is therefore fixed to its neutral value
`0.5` in both training and runtime. This is intentional: it avoids a
train/runtime distribution mismatch and removes the extra per-proposal CLIP
pass. Appearance can be tested later only as a separately trained ablation.

The quality branch is deliberately dual. If a proposed box is rejected, the
runtime uses the original-box quality prediction. Candidate quality can never
be assigned to geometry that was not exported.

## Output and identity contracts

The active `joint_b3_b5_b6v2` ablation:

- keeps B3 at five reliable/diverse views;
- preserves the original BoxFusion OBB basis;
- disables the optional masked-CLIP appearance-memory pass;
- disables the hand-written refit, legacy B5 head, legacy B6 scorer,
  supplemental output, and Soft-NMS;
- changes geometry only when both learned improvement confidence and the
  existing point-support/reprojection gate accept it;
- uses a fixed detector-score blend explicitly recorded by the run wrapper;
- never adds, deletes, or reorders an instance itself.

The existing minimum-extent contract is still applied. The geometry gate
rejects a candidate if it would change whether the original detection survives
that filter, so geometry refinement does not change the final instance count.
`source_indices` and stable IDs remain aligned one-to-one with the original
global detections.

Training collection uses the established `b5v2_memory_observer` profile. It
constructs K=5 memory and writes the exact joint input arrays while preserving
boxes and scores bit-for-bit. It does not load a joint checkpoint.

The versioned diagnostics contain, per observed output, the exact runtime
tensors `joint_points_local [N,5,128,3]`, `joint_point_mask`,
`joint_view_features [N,5,9]`, `joint_view_mask`, `joint_local_boxes [N,6]`,
`joint_quality_features [N,12]`, the local frame centre/basis, and
`joint_input_valid`. The dataset builder consumes these arrays directly; it
does not reconstruct a P=128 training input from the older P=512 diagnostic
cloud.

Dataset format v2 keeps the strict B5 `quality_features` for immutable source
provenance but trains the shared head from runtime-exact
`joint_quality_features`. Historical observer diagnostics used `0.0` for the
disabled legacy refiner after output while the joint input used neutral
`refiner_quality=0.5`. A CPU-only migration accepts exactly that single-column
`0.0 -> 0.5` transition, records its row count in metadata, and rejects every
other feature difference. Newly collected observer diagnostics serialize the
neutral value consistently and therefore need no migrated rows.

## Strict data protocol

Only ScanNet **train** scenes may be used to construct supervision. Both
collection and training scripts:

- reject a scene-list path containing `val`;
- compare scene IDs against the official validation list and reject overlap;
- reject duplicate train scenes;
- require complete prediction/diagnostic pairs;
- use a separate artifact namespace and refuse silent overwrite.

The trainer's `validation-fraction` is a deterministic scene-held-out subset of
ScanNet train. It is not the official ScanNet validation set.

Default locations:

```text
train scenes:
  evaluation/data_util/meta_data/scannetv2_train_b6_100.txt
forbidden validation scenes:
  evaluation/data_util/meta_data/scannetv2_val.txt
train diagnostics:
  diagnostics/joint_b356_k5_p128_observer_train_v1
B5 AP50 supervision:
  datasets/scannet_joint_b356_b5_ap50_train_v1.npz
joint supervision:
  datasets/scannet_joint_b356_k5_p128_train_v1.npz
checkpoint:
  models/scannet_joint_b356_k5_p128_v1.pt
```

Use new paths/tags for a new experiment. Existing artifacts are not replaced.

## Commands

Run these manually and sequentially after other GPU work has finished.

### 1. Collect train-only K=5 diagnostics

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_joint_b356_dev
bash scripts/collect_scannet_joint_b356_train.sh 0,1
```

The default train-frame root is `data/scannet_train`. Override it only with
prepared train RGB-D frames:

```bash
BOXFUSION_SCANNET_FRAMES_ROOT=/path/to/scannet_train_frames \
  bash scripts/collect_scannet_joint_b356_train.sh 0,1
```

An unchanged interrupted collection can be resumed explicitly:

```bash
BOXFUSION_JOINT_B356_ALLOW_COLLECT_RESUME=1 \
  bash scripts/collect_scannet_joint_b356_train.sh 0,1
```

The script rejects mismatched prediction/diagnostic pairs even in resume mode.

### 2. Train on CPU

```bash
bash scripts/train_scannet_joint_b356.sh
```

This command first builds strict K=5 AP50-aware B5 supervision, aligns it to
the exact per-view diagnostic tensors, and then trains the joint checkpoint.
CUDA is hidden for all three steps.

If a previous invocation completed an atomic B5 or joint-dataset stage and
then stopped, resume explicitly:

```bash
BOXFUSION_JOINT_B356_RESUME_TRAIN=1 \
  bash scripts/train_scannet_joint_b356.sh
```

Resume never overwrites a checkpoint. The B5 source is revalidated while
joining every row to the diagnostics; an existing joint dataset is loaded
with the strict schema validator and must contain the SHA-256 of the exact B5
source before training continues.

Common overrides:

```bash
BOXFUSION_JOINT_B356_EPOCHS=100 \
BOXFUSION_JOINT_B356_BATCH_SIZE=32 \
BOXFUSION_JOINT_B356_LR=0.001 \
BOXFUSION_JOINT_B356_CHECKPOINT="$PWD/models/my_joint_v2.pt" \
  bash scripts/train_scannet_joint_b356.sh
```

Choose fresh dataset paths as well when retraining; the script refuses to
overwrite any dataset or checkpoint.

### 3. Fixed ten-scene evaluation first

```bash
bash scripts/run_scannet_joint_b356.sh 0,1
```

Defaults:

```text
scene list: scannetv2_val_ablation10_even.txt
detector blend: 0.40 (explicit wrapper override)
minimum extent: 0.40
profile: joint_b3_b5_b6v2
tag: joint_b356_k5_p128_blend040_ablation10_v1
```

The runtime configuration itself keeps a different default detector blend.
This wrapper explicitly fixes `0.40` to make the comparison with the validated
B6 experiment unambiguous, and prints the effective value before launching.

Do not promote to 100 scenes merely because training converges. First compare
the fixed ten scenes against the exact B6 and identity anchors, including
AP15/AP25/AP50, instance count, accepted geometry changes, threshold-crossing
counts, and runtime.

### 4. Full 100-scene evaluation only after the gate passes

```bash
BOXFUSION_JOINT_B356_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_JOINT_B356_RUN_TAG="joint_b356_k5_p128_blend040_full100_v1" \
  bash scripts/run_scannet_joint_b356.sh 0,1
```

Every tag has independent results, logs, diagnostics, and evaluation output.
The wrapper refuses a non-empty namespace unless resume is explicitly enabled.

## Promotion checks

Before a 100-scene run, verify:

1. observer output is exactly identical to its BoxFusion input;
2. the active joint profile preserves source-index/stable-ID order and count;
3. rejected geometry uses original-branch quality;
4. Q15 ≥ Q25 ≥ Q50 for every output;
5. AP50 and net crossings over IoU 0.50 improve on the fixed ten scenes;
6. AP15/AP25 do not regress materially;
7. one batched joint forward is used at finalization rather than one CUDA call
   per object.

This implementation creates the mechanism and a leakage-safe experiment. It
does not claim a ten-point improvement before the fixed-ten and full-100
evaluations have completed.
