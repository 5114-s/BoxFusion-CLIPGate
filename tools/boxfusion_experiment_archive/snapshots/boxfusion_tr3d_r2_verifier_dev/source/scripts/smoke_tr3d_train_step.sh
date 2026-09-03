#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TR3D_ROOT="$ROOT"
# shellcheck source=scripts/lib/tr3d_launch_common.sh
source "$ROOT/scripts/lib/tr3d_launch_common.sh"
cd "$ROOT"

GPU="${1:-0}"
RUN_TAG="${2:-${BOXFUSION_TR3D_RUN_TAG:-}}"
[[ -n "$RUN_TAG" ]] || tr3d_die \
  "provide a unique run tag: $0 0 tr3d_train_step_smoke_v1"
tr3d_require_tag "$RUN_TAG"
tr3d_parse_gpus "$GPU"
(( ${#TR3D_GPUS[@]} == 1 )) || tr3d_die "train-step smoke uses one GPU"
tr3d_select_env "${BOXFUSION_TR3D_ENV:-$ROOT/.conda/boxfusion-tr3d}"

CONFIG="$ROOT/config/tr3d/tr3d_scannet_foreground_train_step.py"
INIT_CHECKPOINT="$ROOT/models/tr3d_1xb16_scannet-3d-foreground-init.pth"
DATA_ROOT="${BOXFUSION_TR3D_DATA_ROOT:-$ROOT/data/tr3d_scannet}"
ANNOTATION="$DATA_ROOT/annotations/scannet_infos_train_foreground.pkl"
TRAIN_SPLIT="$DATA_ROOT/splits/train.txt"
CONTRACT="$DATA_ROOT/DATASET_CONTRACT.json"
WORK_ROOT="${BOXFUSION_TR3D_SMOKE_WORK_ROOT:-$ROOT/work_dirs/tr3d_smoke/$RUN_TAG}"

tr3d_require_file "$CONFIG"
tr3d_require_file "$INIT_CHECKPOINT"
tr3d_require_file "$CONTRACT"
tr3d_require_file "$ANNOTATION"
tr3d_require_file "$TRAIN_SPLIT"
python "$ROOT/tools/validate_tr3d_experiment.py" \
  --mode full-train \
  --contract "$CONTRACT" \
  --annotation "$ANNOTATION" \
  --expected-split "$TRAIN_SPLIT"
tr3d_conda_run python \
  "$ROOT/tools/verify_tr3d_foreground_checkpoint.py" \
  --output "$INIT_CHECKPOINT"
tr3d_conda_run python \
  "$ROOT/tools/validate_tr3d_training_config.py" \
  --config "$CONFIG" \
  --mode full \
  --allow-disabled-validation
tr3d_check_environment "$CONFIG" "$GPU" 1
tr3d_require_new_root "$WORK_ROOT"

echo "Genuine TR3D T1 one-optimizer-step smoke"
echo "  GPU: $GPU"
echo "  run tag: $RUN_TAG"
echo "  config: $CONFIG"
echo "  init: $INIT_CHECKPOINT"
echo "  work root: $WORK_ROOT"
echo "  batch/workers/repeat: 1/0/1"
echo "  loop: IterBasedTrainLoop max_iters=1; validation disabled"

tr3d_conda_run env CUDA_VISIBLE_DEVICES="$GPU" \
  python "$ROOT/third_party/mmdetection3d/tools/train.py" \
    "$CONFIG" \
    --work-dir "$WORK_ROOT" \
    --cfg-options \
      "randomness.seed=0" \
      "randomness.deterministic=True"

checkpoint="$WORK_ROOT/iter_1.pth"
tr3d_require_file "$checkpoint"
tr3d_conda_run python \
  "$ROOT/tools/smoke_load_tr3d_foreground_checkpoint.py" \
  --config "$ROOT/config/tr3d/tr3d_scannet_foreground.py" \
  --checkpoint "$checkpoint" \
  --device cpu
echo "One real forward/backward/optimizer step completed: $checkpoint"
