#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
DEV="$MAIN/development/final_controls"
SOURCE="$MAIN/development/mvsr_multiview_audit_20261012_source"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
PREPARED_ROOT="$MAIN/data/multiscan_boxfusion_eval50"
SCENE_LIST="$PREPARED_ROOT/scenes.txt"
BASE_CONFIG="$MAIN/reports/multiscan_eval50_calrv1_20261001/configs/recar3d_locked.yaml"
REPLAY_EVIDENCE="$DEV/experiments/multiscan50_novis_20261006/evidence"
RUN_ROOT="$DEV/experiments/multiscan50_mvsr_temporal_patch_20261012"
CONFIG="$RUN_ROOT/config.yaml"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
SCENE_LIMIT="${SCENE_LIMIT:-5}"
mapfile -t ALL_SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
SCENES=("${ALL_SCENES[@]:0:$SCENE_LIMIT}")
mkdir -p "$RUN_ROOT/logs/scenes" "$RUN_ROOT/mplconfig" "$RUN_ROOT/native" \
  "$RUN_ROOT/online" "$RUN_ROOT/diagnostics/online" "$RUN_ROOT/components"
exec 9>"$RUN_ROOT/logs/run.lock"
flock -n 9 || { echo "MultiScan temporal patch is already active" >&2; exit 1; }
exec > >(tee -a "$RUN_ROOT/logs/driver.log") 2>&1
if [[ ! -s "$CONFIG" ]]; then
  "$PYTHON" "$DEV/tools/prepare_mvsr_multiview_audit_config.py" \
    --base "$BASE_CONFIG" --output "$CONFIG" --run-root "$RUN_ROOT" \
    --replay-evidence-root "$REPLAY_EVIDENCE"
fi
validate() {
  local scene="$1" log="$RUN_ROOT/logs/scenes/$1.log"
  [[ -s "$RUN_ROOT/native/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/diagnostics/online/${scene}.json" ]] || return 1
  [[ -s "$RUN_ROOT/components/native_factor_no_visibility_latest/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/components/native_factor_no_visibility_max_ge1/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$log" && "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log"
}
echo "[$(date '+%F %T')] MultiScan MVSR temporal patch started ($SCENE_LIMIT scenes)"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$RUN_ROOT/logs/scenes/$scene.log"
  if validate "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Reusing $scene"
    continue
  fi
  if [[ -e "$log" ]]; then
    stamp="$(date '+%Y%m%d_%H%M%S')"
    mkdir -p "$RUN_ROOT/interrupted/$scene/$stamp"
    mv "$log" "$RUN_ROOT/interrupted/$scene/$stamp/"
  fi
  echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Running $scene"
  (
    cd "$SOURCE"
    CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$RUN_ROOT/mplconfig" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$SOURCE:$MAIN/third_party/WeDetect" \
    timeout 14400 "$PYTHON" demo.py scannet --model-path "$MODEL" \
      --clip_path "$CLIP_MODEL" --class_txt "$CLASS_TXT" --config "$CONFIG" \
      --device cuda:0 --seq "$scene"
  ) > "$log" 2>&1
  validate "$scene" || { echo "Scene validation failed: $scene" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Completed $scene"
done
echo "[$(date '+%F %T')] MultiScan MVSR temporal patch complete"
