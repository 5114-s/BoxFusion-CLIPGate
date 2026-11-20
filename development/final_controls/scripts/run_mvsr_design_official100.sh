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
RUN_TAG="${RUN_TAG:-mvsr_design_20261005}"
SCENE_LIMIT="${SCENE_LIMIT:-100}"
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
[[ "$SCENE_LIMIT" =~ ^[0-9]+$ ]] && (( SCENE_LIMIT >= 1 && SCENE_LIMIT <= 100 )) || {
  echo "SCENE_LIMIT must be in [1,100]" >&2; exit 1;
}
mapfile -t ALL_SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#ALL_SCENES[@]}" -eq 100 ]] || { echo "Expected official100" >&2; exit 1; }
SCENES=("${ALL_SCENES[@]:0:$SCENE_LIMIT}")
mkdir -p "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT" \
  "$RUN_ROOT/native" "$RUN_ROOT/online" "$RUN_ROOT/evidence" \
  "$RUN_ROOT/diagnostics/online" "$COMPONENT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Runner is already active: $RUN_TAG" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

if [[ ! -s "$CONFIG" ]]; then
  "$PYTHON" "$DEV/tools/prepare_mvsr_design_config.py" \
    --base "$BASE_CONFIG" --output "$CONFIG" --run-root "$RUN_ROOT"
fi

COMPONENTS=(
  native_native native_first native_latest native_mean native_top3_mean
  native_max native_ema native_diverse_max native_reliability native_raw_iou_max
  native_factor_no_proposal_score native_factor_no_visibility
  native_factor_no_center_consistency native_factor_no_scale_consistency
  native_factor_no_overlap_consistency births_plr_v1 births_calr_v2
)

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log" component
  [[ -s "$RUN_ROOT/native/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/online/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/diagnostics/online/${scene}.json" ]] || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" ]] || return 1
  [[ -s "$RUN_ROOT/evidence/${scene}.npz" ]] || return 1
  [[ -s "$log" ]] || return 1
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  for component in "${COMPONENTS[@]}"; do
    [[ -s "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || return 1
  done
  "$PYTHON" - "$RUN_ROOT/diagnostics/online/${scene}.json" <<'PY' >/dev/null
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
s=d['state']
assert d['strictly_causal'] and d['online_incremental']
assert not d['scene_end_inference'] and not s['uses_future_frames']
assert s['m2']['support_mode'] == 'max'
assert s['m2']['strength_variant'] == 'full'
assert len(s['m2_factor_controls']) == 5
assert all(v['frames_seen'] == s['frames_seen'] for v in s['m2_factor_controls'].values())
PY
}

echo "[$(date '+%F %T')] MVSR design run started: $RUN_TAG ($SCENE_LIMIT scenes)"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Reusing $scene"
    continue
  fi
  [[ ! -e "$log" ]] || { echo "Invalid partial log exists: $log" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Running $scene"
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
  echo "[$(date '+%F %T')] [$((index+1))/$SCENE_LIMIT] Completed $scene"
done

[[ ! -e "$ARM_ROOT/manifest.json" ]] || { echo "Arm manifest exists" >&2; exit 1; }
"$PYTHON" "$DEV/tools/materialize_mvsr_design_arms.py" \
  --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" \
  --output "$ARM_ROOT" --expected-scenes "$SCENE_LIMIT"

if (( SCENE_LIMIT < 100 )); then
  echo "[$(date '+%F %T')] Smoke run complete; official evaluation skipped"
  exit 0
fi

ARMS=(native only_2d_iou full no_proposal_score no_visibility \
  no_center_consistency no_scale_consistency no_overlap_consistency \
  latest mean ema top3_mean maximum)
for arm in "${ARMS[@]}"; do
  bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
    "${LABEL_PREFIX}_${arm}" "$ARM_ROOT/$arm"
done
"$PYTHON" "$DEV/tools/summarize_mvsr_design.py" \
  --manifest "$ARM_ROOT/manifest.json" --eval-log-root "$EVAL_LOG_ROOT" \
  --label-prefix "$LABEL_PREFIX" --output "$REPORT_ROOT"
echo "[$(date '+%F %T')] MVSR design run complete: $REPORT_ROOT/REPORT.md"
