#!/usr/bin/env bash
set -euo pipefail

DEV=/data/ZhaoX/BoxFusion/development/final_controls
MAIN=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
BASE_CONFIG="$DEV/config/scannet_recar3d_key_controls_official100.yaml"
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
RUN_TAG="${RUN_TAG:-mvsr_novis_temporal_20261006}"
RUN_ROOT="$DEV/experiments/$RUN_TAG"
CONFIG="$RUN_ROOT/config.yaml"
LOG_ROOT="$RUN_ROOT/logs"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
COMPONENT_ROOT="$RUN_ROOT/components"
ARM_ROOT="$RUN_ROOT/arms"
REPORT_ROOT="$RUN_ROOT/report"
MPL_ROOT="$LOG_ROOT/mplconfig"
EVAL_LOG_ROOT="$MAIN/logs/scannet_official100_real_score"
LABEL_PREFIX="recar3d_${RUN_TAG}"

for required in "$PYTHON" "$BASE_CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]] || { echo "Expected official100" >&2; exit 1; }
mkdir -p "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT" \
  "$RUN_ROOT/native" "$RUN_ROOT/online" "$RUN_ROOT/evidence" \
  "$RUN_ROOT/diagnostics/online" "$COMPONENT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Runner is already active: $RUN_TAG" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

if [[ ! -s "$CONFIG" ]]; then
  "$PYTHON" "$DEV/tools/prepare_mvsr_novis_temporal_config.py" \
    --base "$BASE_CONFIG" --output "$CONFIG" --run-root "$RUN_ROOT"
fi

COMPONENTS=(native_native native_factor_no_visibility_latest \
  native_factor_no_visibility_mean native_factor_no_visibility_ema \
  native_factor_no_visibility_top3_mean native_factor_no_visibility_max)

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log" component
  [[ -s "$RUN_ROOT/native/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/online/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/diagnostics/online/${scene}.json" ]] || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" ]] || return 1
  [[ -s "$RUN_ROOT/evidence/${scene}.npz" ]] || return 1
  [[ -s "$log" && "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  for component in "${COMPONENTS[@]}"; do
    [[ -s "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || return 1
  done
  "$PYTHON" - "$RUN_ROOT/diagnostics/online/${scene}.json" <<'PY' >/dev/null
import json, sys
s=json.load(open(sys.argv[1], encoding='utf-8'))['state']
assert s['strictly_causal'] and not s['uses_future_frames']
assert s['m2']['support_mode']=='max'
assert tuple(s['m2_factor_controls']) == ('no_visibility',)
assert s['m2_factor_controls']['no_visibility']['frames_seen'] == s['frames_seen']
PY
}

archive_partial_scene() {
  local scene="$1" stamp archive path
  stamp="$(date '+%Y%m%d_%H%M%S')"
  archive="$RUN_ROOT/interrupted/$scene/$stamp"
  mkdir -p "$archive"
  for path in \
    "$SCENE_LOG_ROOT/$scene.log" \
    "$RUN_ROOT/native/${scene}_boxes.pkl" \
    "$RUN_ROOT/online/${scene}_boxes.pkl" \
    "$RUN_ROOT/diagnostics/online/${scene}.json" \
    "$RUN_ROOT/evidence/${scene}.npz" \
    "$COMPONENT_ROOT/${scene}.json"; do
    if [[ -e "$path" ]]; then
      mkdir -p "$archive/$(dirname "${path#$RUN_ROOT/}")"
      mv "$path" "$archive/${path#$RUN_ROOT/}"
    fi
  done
  while IFS= read -r -d '' path; do
    mkdir -p "$archive/$(dirname "${path#$RUN_ROOT/}")"
    mv "$path" "$archive/${path#$RUN_ROOT/}"
  done < <(find "$COMPONENT_ROOT" -mindepth 2 -maxdepth 2 -type f \
    -name "${scene}_boxes.pkl" -print0)
  echo "Archived interrupted artifacts for $scene to $archive"
}

echo "[$(date '+%F %T')] final no-visibility temporal run started"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/100] Reusing $scene"; continue
  fi
  if [[ -e "$log" ]]; then
    archive_partial_scene "$scene"
  fi
  echo "[$(date '+%F %T')] [$((index+1))/100] Running $scene"
  (
    cd "$DEV"
    CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL_ROOT" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$DEV:$MAIN/third_party/WeDetect" \
    "$PYTHON" demo.py scannet --model-path "$MODEL" --clip_path "$CLIP_MODEL" \
      --class_txt "$CLASS_TXT" --config "$CONFIG" --device cuda:0 --seq "$scene"
  ) > "$log" 2>&1
  validate_scene "$scene" || { echo "Scene validation failed: $scene" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index+1))/100] Completed $scene"
done

if [[ ! -s "$ARM_ROOT/manifest.json" ]]; then
  "$PYTHON" "$DEV/tools/materialize_mvsr_novis_temporal_arms.py" \
    --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" \
    --output "$ARM_ROOT" --expected-scenes 100
else
  echo "Reusing temporal arm manifest: $ARM_ROOT/manifest.json"
fi
for arm in native latest mean ema top3_mean maximum; do
  bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
    "${LABEL_PREFIX}_${arm}" "$ARM_ROOT/$arm"
done
"$PYTHON" "$DEV/tools/summarize_mvsr_novis_temporal.py" \
  --manifest "$ARM_ROOT/manifest.json" --eval-log-root "$EVAL_LOG_ROOT" \
  --label-prefix "$LABEL_PREFIX" --output "$REPORT_ROOT"
echo "[$(date '+%F %T')] final no-visibility temporal run complete: $REPORT_ROOT/REPORT.md"
