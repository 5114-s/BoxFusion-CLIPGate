# B6: IoU-aware multi-view quality ranking

This branch implements a pure score-ranking ablation. It does not change
BoxFusion geometry, add supplemental boxes, run the neural BoxRefiner, or
enable Soft-NMS. Unobserved detections retain their original real score.

## Isolation

Development lives under:

```text
/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev
```

The running full-100 experiment continues to use
`boxfusion_stage3_dev`. B6 scripts derive all prediction, log, diagnostic,
evaluation, dataset, and checkpoint paths from the B6 directory.

## Model

The scorer consumes the fixed 12-feature runtime vector and trains a shared
MLP with four outputs:

```text
predicted continuous 3D IoU
P(IoU >= 0.15)
P(IoU >= 0.25)
P(IoU >= 0.50)
```

Training uses Smooth-L1 for continuous IoU, class-balanced BCE for the three
thresholds, and a monotonic penalty. Runtime projects the probabilities to
`P15 >= P25 >= P50` and combines all four heads with checkpointed non-negative
ranking weights. Checkpoints are strict, versioned, pickle-free NPZ files.

The `box_stability` feature now measures Top-K multi-view lifted-box center and
log-size dispersion. Depth support is log-normalized to avoid saturating for
nearly every confirmed track.

## Data leakage rule

Do not train on `scannetv2_val.txt`, the fixed val-10 list, or diagnostics from
the headline val-100 experiment. Use official ScanNet train scenes. Training
archives store one `scene_ids` value per sample and split complete scenes, so
one scene cannot occur in both train and validation partitions.

Diagnostics should be collected with `quality_observer`: appearance evidence
is recorded, but exported boxes and scores remain unchanged.

The repository includes a deterministic 100-scene train-only subset with one
scan per physical scene:

```text
evaluation/data_util/meta_data/scannetv2_train_b6_100.txt
```

This machine already has the complete extracted train RGB-D data under
`/extra/ZhaoX/scannet_data/scans.sens`. Create the isolated lightweight links:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev
bash scripts/prepare_scannet_b6_train_data.sh
```

The helper does not copy or re-extract images. It links the explicit train-only
subset into `data/scannet_train/<scene>/frames`.

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev
bash scripts/collect_scannet_b6_train_diagnostics.sh 0,1
```

Build the labelled archive and train on CPU:

```bash
bash scripts/train_scannet_b6_quality.sh
```

## Pure B6 fixed-10 evaluation

Wait for any current GPU experiment to finish. Then run:

```bash
bash scripts/run_scannet_b6_quality_only.sh 0,1
```

`quality_only` enforces:

```text
refit=false
box_refiner=false
supplemental_output=false
quality=true
soft_nms=false
apply_to_unobserved=false
```

The default detector blend is zero because detector score is already one of
the learned inputs. It can be ablated without editing code:

```bash
BOXFUSION_B6_DETECTOR_BLEND=0.25 \
BOXFUSION_B6_RUN_TAG=b6_iou_mlp_blend025_ablation10 \
BOXFUSION_QUALITY_CHECKPOINT="$PWD/models/scannet_b6_iou_mlp.npz" \
bash scripts/run_scannet_b6_quality_only.sh 0,1
```

Advance to 100 scenes only if the fixed development experiment reaches:

```text
AP25: at least +1.5
AP50: at least +1.0
AP15: no more than -0.3
box coordinates and box count: identical to the matched B0
```

## Minimum-extent diagnostic

The hard ScanNet output filter is independently testable at 0.30, 0.20, and
0.15 m. This script disables online refinement, so only the threshold changes:

```bash
bash scripts/run_scannet_min_extent_ablation.sh 0,1
```

Treat this as a protocol diagnostic, not as a B6 gain. Report the selected
threshold explicitly in every comparison.
