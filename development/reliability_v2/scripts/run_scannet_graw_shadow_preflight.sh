#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion2
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
CONFIG="$ROOT/config/scannet_graw_shadow_replay_score05.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_graw_e2_preflight3.txt"
CACHE_INDEX="$ROOT/cache/cutr_postfilter_v3/scannet-graw-e2-score05-preflight3-v3-r1/index.json"
PRED_ROOT="$ROOT/results/scannet_graw_shadow_replay_score05"
LOG_ROOT="$ROOT/logs/scannet_graw_shadow_replay_score05"
DIAGNOSTICS_ROOT="$LOG_ROOT/diagnostics"

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "GPU must be a non-negative integer" >&2
  exit 2
fi
if [[ ! -f "$CACHE_INDEX" ]]; then
  echo "Sealed cache index is missing: $CACHE_INDEX" >&2
  exit 1
fi
FINGERPRINT=$("$PYTHON" "$ROOT/tools/proposal_cache_fingerprint.py" \
  from-index --index "$CACHE_INDEX")
export BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$FINGERPRINT"

mkdir -p "$PRED_ROOT" "$LOG_ROOT/scenes" "$LOG_ROOT/mplconfig" "$DIAGNOSTICS_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
  echo "Another Graw-shadow preflight holds $LOG_ROOT/run.lock" >&2
  exit 1
fi

echo "[$(date '+%F %T')] Starting Graw-shadow preflight on GPU $GPU"
echo "[$(date '+%F %T')] Producer fingerprint: $FINGERPRINT"
while IFS= read -r scene || [[ -n "$scene" ]]; do
  prediction="$PRED_ROOT/${scene}_boxes.pkl"
  scene_log="$LOG_ROOT/scenes/${scene}.log"
  observer_json="$DIAGNOSTICS_ROOT/${scene}.observer_tracks.json"
  graw_json="$DIAGNOSTICS_ROOT/${scene}.graw_shadow.json"
  for expected_fresh in "$prediction" "$observer_json" "$graw_json"; do
    if [[ -e "$expected_fresh" ]]; then
      echo "Fresh shadow artifact already exists: $expected_fresh" >&2
      exit 1
    fi
  done
  echo "[$(date '+%F %T')] Running $scene"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
    LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
    "$PYTHON" demo.py scannet \
      --model-path "$ROOT/models/cutr_rgbd.pth" \
      --clip_path "$ROOT/models/open_clip_pytorch_model.bin" \
      --config "$CONFIG" \
      --device cuda \
      --seq "$scene"
  ) >"$scene_log" 2>&1
  for required in "$prediction" "$observer_json" "$graw_json"; do
    if [[ ! -s "$required" ]]; then
      echo "Missing Graw-shadow artifact: $required; see $scene_log" >&2
      exit 1
    fi
  done
  if ! grep -q 'Graw-shadow summary | trace_valid=True' "$scene_log"; then
    echo "Graw-shadow trace is invalid for $scene; see $scene_log" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] Completed $scene"
done < "$SCENE_LIST"

echo "[$(date '+%F %T')] Completed Graw-shadow preflight"
