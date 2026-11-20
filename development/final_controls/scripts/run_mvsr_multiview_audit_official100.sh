#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
CONTROL="$MAIN/development/final_controls"
SOURCE="${SOURCE:-$MAIN/development/mvsr_multiview_audit_20261012_source}"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
BASE_CONFIG="$SOURCE/config/scannet_recar3d_key_controls_official100.yaml"
SCENE_LIST="$MAIN/evaluation/data_util/meta_data/scannetv2_val.txt"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
REPLAY_EVIDENCE="$MAIN/development/final_controls/experiments/mvsr_unified_samepass_20261010/evidence"
RUN_TAG="${RUN_TAG:-mvsr_multiview_audit_20261012}"
RUN_ROOT="$CONTROL/experiments/$RUN_TAG"
CONFIG="$RUN_ROOT/config.yaml"
LOG_ROOT="$RUN_ROOT/logs"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
COMPONENT_ROOT="$RUN_ROOT/components"
ARM_ROOT="$RUN_ROOT/arms"
REPORT_ROOT="$RUN_ROOT/report"
BOOTSTRAP_ROOT="$RUN_ROOT/bootstrap"
MPL_ROOT="$LOG_ROOT/mplconfig"
EVAL_LOG_ROOT="$MAIN/logs/scannet_official100_real_score"
LABEL_PREFIX="recar3d_${RUN_TAG}"

for required in "$PYTHON" "$SOURCE/demo.py" "$BASE_CONFIG" "$SCENE_LIST" \
  "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" "$REPLAY_EVIDENCE"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]] || { echo "Expected official100" >&2; exit 1; }
mkdir -p "$SCENE_LOG_ROOT" "$MPL_ROOT" "$RUN_ROOT/native" \
  "$RUN_ROOT/online" "$RUN_ROOT/diagnostics/online" "$COMPONENT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Runner is already active: $RUN_TAG" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

if [[ ! -s "$CONFIG" ]]; then
  "$PYTHON" "$CONTROL/tools/prepare_mvsr_multiview_audit_config.py" \
    --base "$BASE_CONFIG" --output "$CONFIG" --run-root "$RUN_ROOT" \
    --replay-evidence-root "$REPLAY_EVIDENCE"
fi

REQUIRED_COMPONENTS=(
  native_factor_no_visibility_native
  native_factor_no_visibility_max_ge1
  native_factor_no_visibility_max_ge2
  native_factor_no_visibility_max_ge3
)

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log" component
  [[ -s "$RUN_ROOT/native/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$RUN_ROOT/diagnostics/online/${scene}.json" ]] || return 1
  [[ -s "$RUN_ROOT/native_prefix/${scene}.npz" ]] || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" ]] || return 1
  [[ -s "$log" && "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  for component in "${REQUIRED_COMPONENTS[@]}"; do
    [[ -s "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || return 1
  done
  "$PYTHON" - "$RUN_ROOT/diagnostics/online/${scene}.json" <<'PY' >/dev/null
import json, sys
d=json.load(open(sys.argv[1], encoding='utf-8'))
s=d['state']
assert d['strictly_causal'] and d['online_incremental']
assert s['mvsr_only'] and not s['uses_future_frames']
assert s['m2_factor_controls']['no_visibility']['frames_seen'] == s['frames_seen']
assert d['provider']['replayed']
assert d['native_prefix_keyframes'] == s['frames_seen']
PY
}

archive_partial_scene() {
  local scene="$1" stamp archive path
  stamp="$(date '+%Y%m%d_%H%M%S')"
  archive="$RUN_ROOT/interrupted/$scene/$stamp"
  mkdir -p "$archive"
  for path in "$SCENE_LOG_ROOT/$scene.log" "$RUN_ROOT/native/${scene}_boxes.pkl" \
    "$RUN_ROOT/online/${scene}_boxes.pkl" "$RUN_ROOT/diagnostics/online/${scene}.json" \
    "$COMPONENT_ROOT/${scene}.json"; do
    if [[ -e "$path" ]]; then
      mkdir -p "$archive/$(dirname "${path#$RUN_ROOT/}")"
      mv "$path" "$archive/${path#$RUN_ROOT/}"
    fi
  done
  while IFS= read -r -d '' path; do
    mkdir -p "$archive/$(dirname "${path#$RUN_ROOT/}")"
    mv "$path" "$archive/${path#$RUN_ROOT/}"
  done < <(find "$COMPONENT_ROOT" -mindepth 2 -maxdepth 2 -type f -name "${scene}_boxes.pkl" -print0)
}

echo "[$(date '+%F %T')] MVSR distinct-view audit started"
echo "source=$SOURCE"
echo "evidence_replay=$REPLAY_EVIDENCE"
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

if [[ ! -s "$ARM_ROOT/manifest.json" ]]; then
  "$PYTHON" "$CONTROL/tools/materialize_mvsr_multiview_audit.py" \
    --scene-list "$SCENE_LIST" --components "$COMPONENT_ROOT" \
    --output "$ARM_ROOT" --expected-scenes 100
fi
for arm in native single_view two_view three_view shuffled_negative; do
  bash "$MAIN/scripts/eval_scannet_official100_real_score.sh" \
    "${LABEL_PREFIX}_${arm}" "$ARM_ROOT/$arm"
done
"$PYTHON" "$CONTROL/tools/summarize_mvsr_multiview_audit.py" \
  --manifest "$ARM_ROOT/manifest.json" --eval-log-root "$EVAL_LOG_ROOT" \
  --label-prefix "$LABEL_PREFIX" --output "$REPORT_ROOT"
if [[ ! -s "$BOOTSTRAP_ROOT/results.json" ]]; then
  "$PYTHON" "$MAIN/tools/bootstrap_final_recar_pairs.py" \
    --dataset scannet --scene-list "$SCENE_LIST" --base-root "$ARM_ROOT/native" \
    --treatment "single_view=$ARM_ROOT/single_view" \
    --treatment "two_view=$ARM_ROOT/two_view" \
    --treatment "three_view=$ARM_ROOT/three_view" \
    --treatment "shuffled_negative=$ARM_ROOT/shuffled_negative" \
    --replicates 2000 --seed 20261012 --output "$BOOTSTRAP_ROOT"
fi
echo "[$(date '+%F %T')] MVSR distinct-view audit complete"
