#!/usr/bin/env bash
set -euo pipefail

task_repo_root="/data/ZhaoX/BoxFusion"
task_pipeline_root="$task_repo_root/tools/boxfusion_tr3d_pipeline"
task_python="/home/admin1/miniconda3/envs/boxfusion-online/bin/python"
task_scene_list="$task_pipeline_root/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
task_frames_root="$task_repo_root/upstream_clean/scannet_readme_frames"
task_runner="$task_pipeline_root/scripts/run_boxfusion_sequences.py"

export LD_LIBRARY_PATH="/home/admin1/miniconda3/envs/boxfusion-online/lib"
export MPLCONFIGDIR="/tmp/boxfusion_puf_arb_fixed10_mpl"
export XDG_CACHE_HOME="/tmp/boxfusion_puf_arb_fixed10_xdg"

cd "$task_repo_root"

"$task_python" "$task_runner" \
  --dataset scannet \
  --seq-list "$task_scene_list" \
  --config "$task_pipeline_root/config/scannet_eval.yaml" \
  --model-path "$task_repo_root/models/cutr_rgbd.pth" \
  --clip-path "$task_repo_root/models/open_clip_pytorch_model.bin" \
  --class-txt "$task_repo_root/data/panoptic_categories_nomerge.txt" \
  --class-features "$task_repo_root/data/class_features.pt" \
  --device cuda \
  --gpu 1 \
  --seed 0 \
  --scannet-frames-root "$task_frames_root" \
  --output-dir "$task_pipeline_root/results/puf_lite/fixed10_control_seed0_v1" \
  --log-dir "$task_pipeline_root/logs/puf_lite/fixed10_control_seed0_v1"

"$task_python" "$task_runner" \
  --dataset scannet \
  --seq-list "$task_scene_list" \
  --config "$task_pipeline_root/config/scannet_qim_puf_arbitration_shadow.yaml" \
  --model-path "$task_repo_root/models/cutr_rgbd.pth" \
  --clip-path "$task_repo_root/models/open_clip_pytorch_model.bin" \
  --class-txt "$task_repo_root/data/panoptic_categories_nomerge.txt" \
  --class-features "$task_repo_root/data/class_features.pt" \
  --device cuda \
  --gpu 1 \
  --seed 0 \
  --scannet-frames-root "$task_frames_root" \
  --output-dir "$task_pipeline_root/results/puf_lite/fixed10_arbitration_seed0_v1" \
  --log-dir "$task_pipeline_root/logs/puf_lite/fixed10_arbitration_seed0_v1"
