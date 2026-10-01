# Online refinement quick start

All commands below run from the isolated branch:

```bash
cd /data/ZhaoX/OVM3D-Dett/boxfusion_stage3_dev
```

They read the existing CuTR/CLIP weights and ScanNet frames from
`/data/ZhaoX/BoxFusion`, but write predictions, logs, caches, and diagnostics
under the isolated checkout. They do not change the currently running Stage-2
process.

## 1. CPU validation

```bash
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/admin1/miniconda3/envs/boxfusion2/bin/python -m pytest -q

bash -n scripts/run_scannet_online_refinement.sh
```

The plugin-autoload switch avoids an unrelated obsolete ROS
`launch_testing` pytest plugin in this environment.

## 2. Paired parent-path check

This runs the same score-0.4 CLIP-gate + Top-K configuration while disabling
the entire new controller. It does not require YOLOE:

```bash
BOXFUSION_DISABLE_ONLINE_REFINEMENT=1 \
BOXFUSION_ONLINE_PRED_ROOT="$PWD/results/scannet_online_parent_control" \
BOXFUSION_ONLINE_LOG_ROOT="$PWD/logs/scannet_online_parent_control" \
bash scripts/run_scannet_online_refinement.sh 0
```

Compare this result with the Stage-2 result before attributing any change to
the new modules.

## 3. Accuracy-first Mask-RGBD run

The repository contains the model adapter, not third-party dependencies or
weights. The current `boxfusion2` environment on this machine does not contain
`ultralytics`, so do not modify it while Stage 2 is using it. After Stage 2,
clone the environment and install YOLOE there:

```bash
CONDA_NO_PLUGINS=true /home/admin1/miniconda3/bin/conda create \
  --name boxfusion-online \
  --clone /home/admin1/miniconda3/envs/boxfusion2 \
  --yes
/home/admin1/miniconda3/envs/boxfusion-online/bin/python -m pip install \
  --no-cache-dir ultralytics==8.4.105
```

Use a prompt-free segmentation checkpoint listed in the
[official Ultralytics YOLOE documentation](https://docs.ultralytics.com/models/yoloe/).
The current prompt-free API is `YOLOE("...-seg-pf.pt").predict(...)`; no
prompt vocabulary is required for a PF checkpoint. This checkout uses the
official YOLOE-11S-PF asset:

```bash
mkdir -p models
wget -O models/yoloe-11s-seg-pf.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11s-seg-pf.pt
sha256sum models/yoloe-11s-seg-pf.pt
```

The downloaded file used for the initial smoke test has size `27948751` bytes
and SHA-256
`292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d`.
Supply the local path and the cloned environment explicitly:

```bash
BOXFUSION_ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online \
BOXFUSION_YOLOE_CHECKPOINT=/absolute/path/yoloe-11s-seg-pf.pt \
BOXFUSION_PROPOSAL_INTERVAL=1 \
bash scripts/run_scannet_online_refinement.sh 0
```

Run exactly one scene before a 100-scene ablation:

```bash
BOXFUSION_ONLINE_ABLATION_PROFILE=supplemental_only \
bash scripts/run_single_scene_online_smoke.sh 0 scene0702_01
```

The smoke runner refuses to reuse an existing prediction and requires both a
score-preserving prediction pickle and a pickle-free diagnostics archive.
The checked-in ScanNet config enables both candidate-lifecycle fixes:
`ttl_clock: provider_call` (B1) and `archive_confirmed: true` (B2).

To isolate their effect, keep the `supplemental_only` profile and use separate
output/log directories for these three variants:

```bash
export BOXFUSION_ONLINE_ABLATION_PROFILE=supplemental_only

# Legacy lifecycle control
BOXFUSION_CANDIDATE_TTL_CLOCK=keyframe \
BOXFUSION_ARCHIVE_CONFIRMED_TRACKS=0 \
bash scripts/run_scannet_online_refinement.sh 0

# B1 only
BOXFUSION_CANDIDATE_TTL_CLOCK=provider_call \
BOXFUSION_ARCHIVE_CONFIRMED_TRACKS=0 \
bash scripts/run_scannet_online_refinement.sh 0

# B1+B2
BOXFUSION_CANDIDATE_TTL_CLOCK=provider_call \
BOXFUSION_ARCHIVE_CONFIRMED_TRACKS=1 \
bash scripts/run_scannet_online_refinement.sh 0
```

For each command also set unique
`BOXFUSION_ONLINE_PRED_ROOT`, `BOXFUSION_ONLINE_LOG_ROOT`,
`BOXFUSION_DIAGNOSTICS_ROOT`, and `BOXFUSION_EVAL_ROOT`; otherwise completed
predictions may be reused.

With the provider-call clock, sweep
`BOXFUSION_CANDIDATE_TRACK_TTL=2`, `3`, and `5` on the fixed development
subset.  This value counts missed YOLOE calls, not BoxFusion keyframes.

The fixed B1 conservative-gate experiment keeps TTL=3, disables B2 archive,
and applies the score/projection/global-IoU thresholds `0.25/0.30/0.30`.
It also keeps the ScanNet 0.30-m extent filter active before global
de-duplication.  Its wrapper defaults to the deterministic 10-scene split and
isolated output directories:

```bash
bash scripts/run_scannet_b1_conservative.sh 0,1
```

The projection gate is the proposal-score-weighted mean IoU between the final
3D AABB projected into each stored view and that view's YOLOE 2D box.  It is
not the placeholder `projection_mask_iou` stored while initially lifting an
unmatched proposal.  Use a new `BOXFUSION_B1_RUN_TAG` for every parameter
variant so predictions are never silently reused.

An interrupted full run is resumable with the same scene list and run tag.
The driver skips only non-empty prediction files, reruns unfinished scenes,
and starts evaluation only after the prediction count equals the scene count:

```bash
BOXFUSION_B1_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
BOXFUSION_B1_RUN_TAG="b1_conservative_pj03_g03_s025_full100" \
bash scripts/run_scannet_b1_conservative.sh 0,1
```

If a worker exits, the driver reports its GPU and scene and prints the final
40 lines of that scene log. A message beginning with
`Resolved duplicate fusion stable IDs` is an expected collision-repair
diagnostic, not a failure.

For a lower-latency run, keep the default interval of five BoxFusion
keyframes:

```bash
BOXFUSION_YOLOE_CHECKPOINT=/absolute/path/yoloe-11s-seg-pf.pt \
bash scripts/run_scannet_online_refinement.sh 0
```

Two GPU indices create two independent scene shards:

```bash
BOXFUSION_YOLOE_CHECKPOINT=/absolute/path/yoloe-11s-seg-pf.pt \
bash scripts/run_scannet_online_refinement.sh 0,1
```

This is throughput parallelism across scenes, not a claim that one online
sequence uses two GPUs.

## 4. Oracle analysis

Use this offline tool to measure recall, duplicate/false-positive load, and
the ranking upper bound before deciding how much emphasis to place on
supplemental proposals:

```bash
python tools/analyze_fused_oracle.py \
  --pred-root results/scannet_online_parent_control \
  --gt-root /data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data \
  --scan-root /extra/ZhaoX/scannet_data/scans \
  --scene-list evaluation/data_util/meta_data/scannetv2_val.txt \
  --output results/scannet_online_parent_control/oracle.json
```

This tool reads ground truth and must never be imported by online inference.

## 5. Build training data

First run with:

```yaml
online_refinement:
  diagnostics:
    enabled: true
    dump_track_memory: true
```

Then build a pickle-free training archive. Use a development scene list,
not the final validation set used for the headline number:

```bash
python tools/build_refiner_dataset.py \
  --diagnostics-root results/scannet_online_refinement_diagnostics \
  --scan-root /extra/ZhaoX/scannet_data/scans \
  --gt-root /data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data \
  --scene-list /absolute/path/scannet_development_scenes.txt \
  --min-iou 0.15 \
  --include-negatives \
  --output datasets/box_refiner_development.npz
```

The builder applies each scene's `axisAlignment` to both points and predicted
AABBs before matching them with ScanNet ground truth.

## 6. Train the two light heads

```bash
CUDA_VISIBLE_DEVICES='' python tools/train_box_refiner.py \
  --input datasets/box_refiner_development.npz \
  --output models/box_refiner_development.pt \
  --epochs 40 \
  --batch-size 32

python tools/train_quality_calibrator.py \
  --input datasets/box_refiner_development.npz \
  --output models/quality_linear_development.npz \
  --target-kind iou
```

Run both learned heads:

```bash
BOXFUSION_YOLOE_CHECKPOINT=/absolute/path/yoloe-11s-seg-pf.pt \
BOXFUSION_REFINER_CHECKPOINT="$PWD/models/box_refiner_development.pt" \
BOXFUSION_QUALITY_CHECKPOINT="$PWD/models/quality_linear_development.npz" \
BOXFUSION_QUALITY_MODE=linear \
bash scripts/run_scannet_online_refinement.sh 0
```

Checkpoints are loaded strictly: model architecture and the ordered 12-feature
quality schema must match exactly.

## 7. Required reporting

Report at least:

- AP at IoU 0.15, 0.25, and 0.50 with real detector scores;
- the paired disabled-controller result;
- proposal-only, memory/refit, learned BoxRefiner, and quality/Soft-NMS
  ablations;
- class-agnostic recall and oracle-ranked AP;
- end-to-end single-GPU FPS including mask inference and cache misses;
- GPU memory and the proposal scheduling interval.

No module in this route guarantees a 10- or 15-point AP gain. The code makes
the hypotheses testable without silently changing the evaluation protocol.
