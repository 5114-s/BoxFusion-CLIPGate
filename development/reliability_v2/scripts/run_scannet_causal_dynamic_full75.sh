#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
GPU="${BOXFUSION_GPU:-0}"
CPU_THREADS="${BOXFUSION_CPU_THREADS:-1}"
CONFIG="$ROOT/config/scannet_dyn_causal_dynamic_active.yaml"
MODEL="${BOXFUSION_MODEL:-$ROOT/models/cutr_rgbd.pth}"
MANIFEST="${BOXFUSION_DYNAMIC_MANIFEST:-$ROOT/data_dyn/manifest.json}"
LOG_ROOT="${BOXFUSION_DYNAMIC_LOG_ROOT:-$ROOT/logs/causal_dynamic/full75}"
PERSISTENT="$ROOT/results/scannet_dyn_causal_dynamic_persistent"
CURRENT="$ROOT/results/scannet_dyn_causal_dynamic_current"
EVENTS="$ROOT/diagnostics/dynamic_objects/events"

SCENE_LIST="$(mktemp /tmp/boxfusion_dynamic_scenes.XXXXXX.txt)"
trap 'rm -f -- "$SCENE_LIST"' EXIT

"$PYTHON" -c 'import json,pathlib,sys; value=json.loads(pathlib.Path(sys.argv[1]).read_text()); pathlib.Path(sys.argv[2]).write_text("".join(f"{key}\n" for key in sorted(value)))' "$MANIFEST" "$SCENE_LIST"

ENV_LIB=/home/admin1/miniconda3/envs/boxfusion2/lib
CUDA_VISIBLE_DEVICES="$GPU" LD_LIBRARY_PATH="$ENV_LIB:$ENV_LIB/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" MPLCONFIGDIR=/tmp/boxfusion_dynamic_mpl \
  OPENBLAS_NUM_THREADS="$CPU_THREADS" OMP_NUM_THREADS="$CPU_THREADS" \
  MKL_NUM_THREADS="$CPU_THREADS" \
  "$PYTHON" "$ROOT/scripts/run_boxfusion_sequences.py" \
    --dataset scannet \
    --seq-list "$SCENE_LIST" \
    --config "$CONFIG" \
    --model-path "$MODEL" \
    --device cuda \
    --gpu "$GPU" \
    --log-dir "$LOG_ROOT"

"$PYTHON" "$ROOT/tools/validate_dynamic_run_coverage.py" \
  --manifest "$MANIFEST" \
  --persistent-root "$PERSISTENT" \
  --current-root "$CURRENT" \
  --event-root "$EVENTS"
