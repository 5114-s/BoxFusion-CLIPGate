# Lightweight online TR3D fusion ablation

This is an isolated cumulative ablation rooted at the frozen
`B6 + Selective Boxer G0 + terminal R3` anchor.  It never edits an anchor
prediction.  Supplemental rows are produced only after a train-only policy
authorizes the corresponding stage and are always scored below every anchor.

## Stages

| Stage | Only newly introduced component |
|---|---|
| L1 | OV-SCAN-style RGB-D visibility quality |
| L2 | Zoo3D-style quality/diversity Top-K views |
| L3 | latest-only asynchronous incremental TR3D worker |
| L4 | SMOV3D-style free-space contradiction veto |
| L5 | InsFusion-style raw/fused geometry choice |
| L6 | source-aware low-score append ranking |

CLIP labels/features are not modified.  L1/L2 use synchronous TR3D so the
effect of asynchrony first appears at L3.  L3 keeps one in-flight request and
at most one newest pending snapshot; stale pending snapshots are replaced.

## Required protocol per stage

Use a new tag for every rerun.  Example for L6:

```bash
cd /data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline

# 1. Collect train-only diagnostics. No evaluation/validation GT is read.
BOXFUSION_ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online \
BOXFUSION_LIGHTWEIGHT_TRAIN_TAG=tr3d_lightweight_l6_train100_v1 \
bash scripts/collect_scannet_tr3d_lightweight_train.sh 6 0,1

# 2. Build labels from ScanNet train only and train the novelty gate.
BOXFUSION_LIGHTWEIGHT_TRAIN_TAG=tr3d_lightweight_l6_train100_v1 \
bash scripts/train_scannet_tr3d_lightweight_gate.sh 6

# 3. Fixed 10-scene validation ablation.
BOXFUSION_ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online \
BOXFUSION_LIGHTWEIGHT_ACTIVE_TAG=tr3d_lightweight_l6_fixed10_v1 \
bash scripts/run_scannet_tr3d_lightweight_active.sh 6 0,1
```

Do not run step 3 if step 2 prints `activation_authorized: False`.

If fixed10 improves all three AP thresholds without a material runtime loss,
repeat step 3 with a new tag and the full validation list:

```bash
BOXFUSION_ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online \
BOXFUSION_LIGHTWEIGHT_ACTIVE_TAG=tr3d_lightweight_l6_full100_v1 \
BOXFUSION_LIGHTWEIGHT_ACTIVE_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
bash scripts/run_scannet_tr3d_lightweight_active.sh 6 0,1
```

For a strict cumulative ablation, execute L1 through L6 in order, giving each
stage its own train collection, policy, fixed10 namespace, and then compare
against the same-run terminal-R3 anchor printed by the runner.

## Runtime interpretation

The worker's model call is live for the incremental branch, but the current
parent runner still replays CuTR proposals and terminal-p100 R3.  Therefore
its reported FPS is suitable for paired relative comparisons, not a claim of
fully live end-to-end FPS.  L3 reports submitted/completed/replaced/dropped
requests and provider wall time in each diagnostic.

## Smoke verification (2026-08-12)

`scene0277_00` completed with 45 keyframes, 9 submitted/completed requests,
0 stale replacements, 303 tracks and 253 confirmed tracks.  Depth evidence
was present, and raw/fused choice selected 59 fused and 194 raw geometries.
The compatibility materializer appended one low-score candidate to seven
unaltered anchors; the identity audit reported `ok=true`.  This smoke used the
older novelty policy only to test interfaces and is not an accuracy result.
