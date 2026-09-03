#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_SPEC="${1:-0}"
ENV_REF="${BOXFUSION_TR3D_ENV:-$ROOT/.conda/boxfusion-tr3d}"
CONFIG="${BOXFUSION_TR3D_CONFIG:-$ROOT/config/tr3d/tr3d_scannet_foreground_official_val.py}"
CHECKPOINT="${BOXFUSION_TR3D_CHECKPOINT:-}"
MANIFEST="${BOXFUSION_TR3D_INPUT_MANIFEST:-$ROOT/data/tr3d_scannet/scene_manifest.jsonl}"
SCENE_LIST="${BOXFUSION_TR3D_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
CACHE_ROOT="${BOXFUSION_TR3D_CACHE_ROOT:-$ROOT/cache/tr3d_residual/official_val}"
RUN_TAG="${BOXFUSION_TR3D_RUN_TAG:-tr3d_residual_official_val_v1}"
PREFIX_ID="${BOXFUSION_TR3D_PREFIX_ID:-full}"
SCORE_THRESHOLD="${BOXFUSION_TR3D_SCORE_THRESHOLD:-0.01}"
MAX_PROPOSALS="${BOXFUSION_TR3D_MAX_PROPOSALS:-1000}"
VOXEL_SIZE="${BOXFUSION_TR3D_VOXEL_SIZE:-0.01}"
LOG_ROOT="${BOXFUSION_TR3D_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"

if [[ -z "$CHECKPOINT" ]]; then
  echo "Set BOXFUSION_TR3D_CHECKPOINT to the frozen one-class TR3D checkpoint" >&2
  exit 2
fi
for path in "$CONFIG" "$CHECKPOINT" "$MANIFEST" "$SCENE_LIST"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
done

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "At least one GPU is required" >&2
  exit 2
fi
mkdir -p "$LOG_ROOT"

if [[ "$ENV_REF" == */* ]]; then
  CONDA_SELECTOR=(-p "$ENV_REF")
else
  CONDA_SELECTOR=(-n "$ENV_REF")
fi

echo "Genuine TR3D immutable-cache inference"
echo "  GPUs: $GPU_SPEC; workers: ${#GPUS[@]}"
echo "  environment: $ENV_REF"
echo "  config: $CONFIG"
echo "  checkpoint: $CHECKPOINT"
echo "  manifest: $MANIFEST"
echo "  scenes: $SCENE_LIST"
echo "  prefix: $PREFIX_ID"
echo "  cache: $CACHE_ROOT"
echo "  logs: $LOG_ROOT"

pids=()
for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  report="$LOG_ROOT/export_shard_${index}.json"
  log="$LOG_ROOT/worker_${index}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
    conda run --no-capture-output "${CONDA_SELECTOR[@]}" \
    env PYTHONNOUSERSITE=1 \
    python "$ROOT/tools/run_tr3d_cache_inference.py" \
      --config "$CONFIG" \
      --checkpoint "$CHECKPOINT" \
      --cache-root "$CACHE_ROOT" \
      --input-manifest "$MANIFEST" \
      --scene-list "$SCENE_LIST" \
      --prefix-id "$PREFIX_ID" \
      --device cuda:0 \
      --score-threshold "$SCORE_THRESHOLD" \
      --max-proposals "$MAX_PROPOSALS" \
      --voxel-size "$VOXEL_SIZE" \
      --shard-index "$index" \
      --num-shards "${#GPUS[@]}" \
      --resume \
      --report "$report" >"$log" 2>&1 &
  pids+=("$!")
  echo "  launched shard $index on GPU $gpu (pid=${pids[$index]})"
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failed=1
    echo "TR3D shard $index failed; inspect $LOG_ROOT/worker_${index}.log" >&2
  fi
done
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
echo "All TR3D cache shards completed"
