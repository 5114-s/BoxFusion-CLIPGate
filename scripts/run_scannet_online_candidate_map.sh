#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 SCENE_ID [CONFIG]" >&2
  exit 2
fi

SCENE_ID="$1"
CONFIG_PATH="${2:-config/scannet_t05_boxer_online_candidate_map.yaml}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON_BIN="$ENV_ROOT/bin/python"
GPU_PAIR="${CUDA_VISIBLE_DEVICES:-0,1}"

export CUDA_VISIBLE_DEVICES="$GPU_PAIR"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/boxfusion_online_candidate_map_mpl}"
export PYTHONPATH="$(pwd):$(pwd)/third_party/WeDetect${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$MPLCONFIGDIR"

exec "$PYTHON_BIN" demo.py scannet \
  --model-path models/cutr_rgbd.pth \
  --config "$CONFIG_PATH" \
  --device cuda:0 \
  --seq "$SCENE_ID"
