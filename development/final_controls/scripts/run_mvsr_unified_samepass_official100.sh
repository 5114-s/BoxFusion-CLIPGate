#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
CONTROL="$MAIN/development/final_controls"
SOURCE="${SOURCE:-$MAIN/development/mvsr_unified_samepass_20261010_source}"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
BASE_CONFIG="$SOURCE/config/scannet_recar3d_key_controls_official100.yaml"
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
RUN_TAG="${RUN_TAG:-mvsr_unified_samepass_20261010}"
RUN_ROOT="$CONTROL/experiments/$RUN_TAG"
CONFIG="$RUN_ROOT/config.yaml"
LOG_ROOT="$RUN_ROOT/logs"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
COMPONENT_ROOT="$RUN_ROOT/components"
DESIGN_ARM_ROOT="$RUN_ROOT/design_arms"
TEMPORAL_ARM_ROOT="$RUN_ROOT/temporal_arms"
DESIGN_REPORT_ROOT="$RUN_ROOT/design_report"
TEMPORAL_REPORT_ROOT="$RUN_ROOT/temporal_report"
MPL_ROOT="$LOG_ROOT/mplconfig"
EVAL_LOG_ROOT="$MAIN/logs/scannet_official100_real_score"
LABEL_PREFIX="recar3d_${RUN_TAG}"

for required in "$PYTHON" "$SOURCE/demo.py" "$BASE_CONFIG" "$SCENE_LIST" \
  "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" "$SOURCE/SOURCE_MANIFEST.sha256"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]] || { echo "Expected official100" >&2; exit 1; }
mkdir -p "$SCENE_LOG_ROOT" "$MPL_ROOT" "$RUN_ROOT/native" \
  "$RUN_ROOT/online" "$RUN_ROOT/evidence" "$RUN_ROOT/diagnostics/online" \
  "$COMPONENT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Runner is already active: $RUN_TAG" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

if [[ ! -s "$CONFIG" ]]; then
  "$PYTHON" "$CONTROL/tools/prepare_mvsr_design_config.py" \
    --base "$BASE_CONFIG" --output "$CONFIG" --run-root "$RUN_ROOT"
fi

BASE_COMPONENTS=(
  native_native native_first native_latest native_mean native_top3_mean
  native_max native_ema native_diverse_max native_reliability native_raw_iou_max
  native_factor_no_proposal_score native_factor_no_visibility
  native_factor_no_center_consistency native_factor_no_scale_consistency
  native_factor_no_overlap_consistency births_plr_v1 births_calr_v2
)
TEMPORAL_MODES=(native latest mean ema top3_mean max)
FACTOR_VARIANTS=(no_proposal_score no_visibility no_center_consistency \
  no_scale_consistency no_overlap_consistency)

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log" component variant mode
  [[ -s "$RUN_ROOT/native/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/online/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/diagnostics/online/${scene}.json" ]] || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" ]] || return 1
  [[ -s "$RUN_ROOT/evidence/${scene}.npz" ]] || return 1
  [[ -s "$log" && "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  for component in "${BASE_COMPONENTS[@]}"; do
    [[ -s "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || return 1
  done
  for variant in "${FACTOR_VARIANTS[@]}"; do
    for mode in "${TEMPORAL_MODES[@]}"; do
      [[ -s "$COMPONENT_ROOT/native_factor_${variant}_${mode}/${scene}_boxes.pkl" ]] || return 1
    done
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
}

echo "[$(date '+%F %T')] unified same-pass MVSR run started"
echo "source=$SOURCE"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/100] Reusing $scene"
    continue
  fi
  [[ ! -e "$log" ]] || archive_partial_scene "$scene"
  echo "[$(date '+%F %T')] [$((index+1))/100] Running $scene"
  (
    cd "$SOURCE"
    CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL_ROOT" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$SOURCE:$MAIN/third_party/WeDetect" \
    "$PYTHON" demo.py scannet --model-path "$MODEL" --clip_path "$CLIP_MODEL" \
      --class_txt "$CLASS_TXT" --config "$CONFIG" --device cuda:0 --seq "$scene"
  ) > "$log" 2>&1
  validate_scene "$scene" || { echo "Scene validation failed: $scene" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index+1))/100] Completed $scene"
done

if [[ ! -s "$DESIGN_ARM_ROOT/manifest.json" ]]; then
  "$PYTHON" "$CONTROL/tools/materialize_mvsr_design_arms.py" \
    --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" \
    --output "$DESIGN_ARM_ROOT" --expected-scenes 100
fi
if [[ ! -s "$TEMPORAL_ARM_ROOT/manifest.json" ]]; then
  "$PYTHON" "$CONTROL/tools/materialize_mvsr_novis_temporal_arms.py" \
    --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" \
    --output "$TEMPORAL_ARM_ROOT" --expected-scenes 100
fi

DESIGN_ARMS=(native only_2d_iou full no_proposal_score no_visibility \
  no_center_consistency no_scale_consistency no_overlap_consistency \
  latest mean ema top3_mean maximum)
for arm in "${DESIGN_ARMS[@]}"; do
  bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
    "${LABEL_PREFIX}_design_${arm}" "$DESIGN_ARM_ROOT/$arm"
done
for arm in native latest mean ema top3_mean maximum; do
  bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
    "${LABEL_PREFIX}_temporal_${arm}" "$TEMPORAL_ARM_ROOT/$arm"
done

"$PYTHON" "$CONTROL/tools/summarize_mvsr_design.py" \
  --manifest "$DESIGN_ARM_ROOT/manifest.json" --eval-log-root "$EVAL_LOG_ROOT" \
  --label-prefix "${LABEL_PREFIX}_design" --output "$DESIGN_REPORT_ROOT"
"$PYTHON" "$CONTROL/tools/summarize_mvsr_novis_temporal.py" \
  --manifest "$TEMPORAL_ARM_ROOT/manifest.json" --eval-log-root "$EVAL_LOG_ROOT" \
  --label-prefix "${LABEL_PREFIX}_temporal" --output "$TEMPORAL_REPORT_ROOT"
echo "[$(date '+%F %T')] unified same-pass MVSR run complete"

