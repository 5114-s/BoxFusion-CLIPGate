#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TR3D_ROOT="$ROOT"
# shellcheck source=scripts/lib/tr3d_launch_common.sh
source "$ROOT/scripts/lib/tr3d_launch_common.sh"
cd "$ROOT"

GPU_SPEC="${1:-0}"
RUN_TAG="${2:-${BOXFUSION_TR3D_RUN_TAG:-}}"
[[ -n "$RUN_TAG" ]] || tr3d_die \
  "provide a unique run tag: $0 0,1 tr3d_fg_prefix_v1"
tr3d_require_tag "$RUN_TAG"
tr3d_parse_gpus "$GPU_SPEC"
tr3d_select_env "${BOXFUSION_TR3D_ENV:-$ROOT/.conda/boxfusion-tr3d}"

CONFIG="${BOXFUSION_TR3D_CONFIG:-$ROOT/config/tr3d/tr3d_scannet_foreground_prefix.py}"
DATA_ROOT="${BOXFUSION_TR3D_DATA_ROOT:-$ROOT/data/tr3d_scannet}"
ANNOTATION="$DATA_ROOT/annotations/scannet_infos_prefix_train_foreground.pkl"
TRAIN_SPLIT="$DATA_ROOT/splits/train.txt"
CONTRACT="$DATA_ROOT/DATASET_CONTRACT.json"
BASE_CHECKPOINT="${BOXFUSION_TR3D_BASE_CHECKPOINT:-}"
WORK_ROOT="${BOXFUSION_TR3D_WORK_ROOT:-$ROOT/work_dirs/tr3d/$RUN_TAG}"
RESUME="${BOXFUSION_TR3D_RESUME:-0}"
if [[ -n "${BOXFUSION_TR3D_BATCH_PER_GPU:-}" ]]; then
  BATCH_SIZE="$BOXFUSION_TR3D_BATCH_PER_GPU"
else
  (( 16 % ${#TR3D_GPUS[@]} == 0 )) || tr3d_die \
    "reference global batch 16 is not divisible by world size; set BOXFUSION_TR3D_BATCH_PER_GPU"
  BATCH_SIZE="$((16 / ${#TR3D_GPUS[@]}))"
fi
NUM_WORKERS="${BOXFUSION_TR3D_NUM_WORKERS:-2}"
PORT="${BOXFUSION_TR3D_PORT:-$((24000 + $$ % 10000))}"
AMP="${BOXFUSION_TR3D_AMP:-1}"

tr3d_require_file "$CONFIG"
tr3d_require_file "$CONTRACT"
tr3d_require_file "$ANNOTATION"
tr3d_require_file "$TRAIN_SPLIT"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] \
  || tr3d_die "BOXFUSION_TR3D_RESUME must be 0 or 1"
if [[ "$RESUME" == "0" ]]; then
  [[ -n "$BASE_CHECKPOINT" ]] || tr3d_die \
    "set BOXFUSION_TR3D_BASE_CHECKPOINT to a frozen T1 full-scene checkpoint"
  tr3d_require_file "$BASE_CHECKPOINT"
fi
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || tr3d_die "invalid batch size"
GLOBAL_BATCH="$((BATCH_SIZE * ${#TR3D_GPUS[@]}))"
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || tr3d_die "invalid worker count"

python "$ROOT/tools/validate_tr3d_experiment.py" \
  --mode prefix-train \
  --contract "$CONTRACT" \
  --annotation "$ANNOTATION" \
  --expected-split "$TRAIN_SPLIT"
tr3d_conda_run python \
  "$ROOT/tools/validate_tr3d_training_config.py" \
  --config "$CONFIG" \
  --mode prefix
tr3d_check_environment "$CONFIG" "${TR3D_GPUS[0]}" 1
tr3d_prepare_unique_root "$WORK_ROOT" "$RESUME"
if [[ "$RESUME" == "1" ]]; then
  tr3d_require_file "$WORK_ROOT/last_checkpoint"
  IFS= read -r RESUME_CHECKPOINT <"$WORK_ROOT/last_checkpoint"
  [[ "$RESUME_CHECKPOINT" == /* ]] \
    || RESUME_CHECKPOINT="$WORK_ROOT/$RESUME_CHECKPOINT"
  tr3d_require_file "$RESUME_CHECKPOINT"
  RESUME_CHECKPOINT="$(readlink -f "$RESUME_CHECKPOINT")"
  WORK_ROOT_REAL="$(readlink -f "$WORK_ROOT")"
  [[ "$RESUME_CHECKPOINT" == "$WORK_ROOT_REAL/"* ]] || tr3d_die \
    "last_checkpoint escapes the selected work root: $RESUME_CHECKPOINT"
  tr3d_conda_run python \
    "$ROOT/tools/smoke_load_tr3d_foreground_checkpoint.py" \
    --config "$ROOT/config/tr3d/tr3d_scannet_foreground.py" \
    --checkpoint "$RESUME_CHECKPOINT" \
    --device cpu
fi

resume_args=()
cfg_options=(
  "train_dataloader.batch_size=$BATCH_SIZE"
  "train_dataloader.num_workers=$NUM_WORKERS"
  "randomness.seed=0"
  "randomness.deterministic=True"
)
if [[ "$RESUME" == "1" ]]; then
  resume_args=(--resume auto)
else
  cfg_options+=("load_from=$BASE_CHECKPOINT")
fi
amp_args=()
[[ "$AMP" == "1" ]] && amp_args=(--amp)

echo "Genuine TR3D T1 trajectory-prefix fine-tuning"
echo "  run tag: $RUN_TAG"
echo "  GPUs: $TR3D_GPU_SPEC (${#TR3D_GPUS[@]} processes)"
echo "  config: $CONFIG"
echo "  annotation: $ANNOTATION"
echo "  base checkpoint: ${BASE_CHECKPOINT:-resume-from-work-root}"
echo "  work root: $WORK_ROOT"
echo "  resume: $RESUME; AMP: $AMP"
echo "  per-GPU/global batch: $BATCH_SIZE/$GLOBAL_BATCH; workers: $NUM_WORKERS"
echo "  official ScanNet val is forbidden for training/checkpoint selection"

train=(
  "$ROOT/third_party/mmdetection3d/tools/train.py"
  "$CONFIG"
  --work-dir "$WORK_ROOT"
  "${amp_args[@]}"
  "${resume_args[@]}"
  --cfg-options "${cfg_options[@]}"
)
if (( ${#TR3D_GPUS[@]} == 1 )); then
  tr3d_conda_run env CUDA_VISIBLE_DEVICES="$TR3D_GPU_SPEC" \
    python "${train[@]}"
else
  tr3d_conda_run env CUDA_VISIBLE_DEVICES="$TR3D_GPU_SPEC" \
    python -m torch.distributed.run \
      --nproc_per_node="${#TR3D_GPUS[@]}" \
      --master_port="$PORT" \
      "${train[@]}" \
      --launcher pytorch
fi
