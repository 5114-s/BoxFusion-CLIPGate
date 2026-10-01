#!/usr/bin/env bash
set -euo pipefail

DEV=/data/ZhaoX/BoxFusion/development/final_controls
PIPE=/data/ZhaoX/BoxFusion/development/final_pipeline
MAIN=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$DEV/config/ca1m_recar3d_final107.yaml"
SCENE_LIST="$MAIN/tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt"
MODEL="$MAIN/models/cutr_rgbd.pth"
CLIP_MODEL="$MAIN/models/open_clip_pytorch_model.bin"
CLASS_TXT="$MAIN/data/panoptic_categories_nomerge.txt"
DATA_ROOT=/extra/ZhaoX/boxfusion_ca1m
NATIVE_ROOT="$DEV/results/ca1m_recar3d_final107_native"
ONLINE_ROOT="$DEV/results/ca1m_recar3d_final107"
DIAGNOSTICS_ROOT="$DEV/diagnostics/ca1m_recar3d_final107"
COMPONENT_ROOT="$DEV/results/ca1m_recar3d_final107_components"
FACTORIAL_ROOT="$PIPE/results/ca1m_recar3d_final107_factorial"
REPORT_ROOT="$PIPE/reports/ca1m_recar3d_final107"
LOG_ROOT="$PIPE/logs/ca1m_recar3d_final107"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"
FINGERPRINT_FILE="$LOG_ROOT/source.fingerprint"

for required in "$PYTHON" "$CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  bd5f3fc66168114048a1b12addc45949c8f54f9c016b921bacfb6fe9e3e7dc2f ]] || {
  echo "CA-1M-107 scene-list hash mismatch" >&2; exit 1;
}
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 107 ]] || { echo "Expected 107 scenes" >&2; exit 1; }
mkdir -p "$NATIVE_ROOT" "$ONLINE_ROOT" "$DIAGNOSTICS_ROOT" "$COMPONENT_ROOT" \
  "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "CA-1M final runner is active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

fingerprint() {
  sha256sum "$CONFIG" "$DEV/demo.py" "$PIPE/run_ca1m_final107.sh" \
    "$PIPE/materialize_factorial.py" \
    "$DEV/boxfusion/reliability_candidate_map.py" \
    "$DEV/boxfusion/native_reliability_reranker.py" \
    "$DEV/boxfusion/causal_reliability.py" \
    "$DEV/boxfusion/m1_anchor_reliability_online.py" \
    "$DEV/boxfusion/m1_anchor_online.py" \
    "$DEV/boxfusion/online_candidate_map.py" \
    "$DEV/boxfusion/online_candidate_runtime.py" \
    "$DEV/boxfusion/box_manager.py" "$DEV/boxfusion/instances.py" \
    "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" "$SCENE_LIST" | sha256sum | awk '{print $1}'
}
RUN_FINGERPRINT="$(fingerprint)"
if [[ -s "$FINGERPRINT_FILE" ]]; then
  [[ "$(tr -d '\n' < "$FINGERPRINT_FILE")" == "$RUN_FINGERPRINT" ]] || {
    echo "CA-1M source/config fingerprint changed" >&2; exit 1;
  }
else
  printf '%s\n' "$RUN_FINGERPRINT" > "$FINGERPRINT_FILE"
fi

COMPONENTS=(native_native native_first native_mean native_max native_ema \
  native_diverse_max native_reliability births_plr_v1 births_calr_v1 births_calr_v2)

validate_scene() {
  local scene="$1" log="$SCENE_LOG_ROOT/$1.log"
  [[ -s "$NATIVE_ROOT/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$ONLINE_ROOT/${scene}_boxes.pkl" ]] || return 1
  [[ -s "$DIAGNOSTICS_ROOT/${scene}.json" ]] || return 1
  [[ -s "$COMPONENT_ROOT/${scene}.json" ]] || return 1
  [[ -s "$log" ]] || return 1
  [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
  local component
  for component in "${COMPONENTS[@]}"; do
    [[ -s "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || return 1
  done
  "$PYTHON" - "$DIAGNOSTICS_ROOT/${scene}.json" "$COMPONENT_ROOT/${scene}.json" <<'PY' >/dev/null
import json, sys
d=json.load(open(sys.argv[1],encoding='utf-8')); c=json.load(open(sys.argv[2],encoding='utf-8'))
s=d['state']
assert d['strictly_causal'] is True and d['online_incremental'] is True
assert d['scene_end_inference'] is False and s['uses_future_frames'] is False
assert s['m1p']['use_children'] is False and s['m1a_control'] is not None
assert s['m2']['support_mode'] == 'max'
assert c['single_online_pass'] is True and c['future_frames_used'] is False
assert len(c['counts']) == 10
PY
}

echo "[$(date '+%F %T')] CA-1M final107 started fingerprint=$RUN_FINGERPRINT"
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"; log="$SCENE_LOG_ROOT/$scene.log"
  [[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || { echo "Source changed" >&2; exit 1; }
  if validate_scene "$scene"; then
    echo "[$(date '+%F %T')] [$((index+1))/107] Reusing $scene"; continue
  fi
  for stale in "$NATIVE_ROOT/${scene}_boxes.pkl" "$ONLINE_ROOT/${scene}_boxes.pkl" \
    "$DIAGNOSTICS_ROOT/${scene}.json" "$COMPONENT_ROOT/${scene}.json" "$log"; do
    [[ ! -e "$stale" ]] || { echo "Invalid partial artifact: $stale" >&2; exit 1; }
  done
  for component in "${COMPONENTS[@]}"; do
    [[ ! -e "$COMPONENT_ROOT/$component/${scene}_boxes.pkl" ]] || {
      echo "Invalid partial component: $component/$scene" >&2; exit 1;
    }
  done
  echo "[$(date '+%F %T')] [$((index+1))/107] Running $scene"
  (
    cd "$DEV"
    CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
    MPLCONFIGDIR="$MPL_ROOT" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    PYTHONPATH="$DEV:$MAIN/third_party/WeDetect" \
    "$PYTHON" demo.py CA1M --model-path "$MODEL" --clip_path "$CLIP_MODEL" \
      --class_txt "$CLASS_TXT" --config "$CONFIG" --device cuda:0 --seq "$scene"
  ) > "$log" 2>&1
  validate_scene "$scene" || { echo "Scene validation failed: $scene" >&2; exit 1; }
  echo "[$(date '+%F %T')] [$((index+1))/107] Completed $scene"
done

[[ "$(fingerprint)" == "$RUN_FINGERPRINT" ]] || { echo "Source changed" >&2; exit 1; }
[[ ! -e "$FACTORIAL_ROOT/manifest.json" ]] || { echo "Factorial exists" >&2; exit 1; }
"$PYTHON" "$PIPE/materialize_factorial.py" \
  --scene-list "$SCENE_LIST" --expected-scenes 107 \
  --components "$COMPONENT_ROOT" --output "$FACTORIAL_ROOT"
"$PYTHON" "$MAIN/tools/evaluate_ca1m_online_factorial.py" \
  --data-root "$DATA_ROOT" --scene-list "$SCENE_LIST" \
  --factorial-root "$FACTORIAL_ROOT" --output "$REPORT_ROOT" \
  2>&1 | tee "$LOG_ROOT/evaluate_factorial.log"
sha256sum "$CONFIG" "$SCENE_LIST" "$FACTORIAL_ROOT/manifest.json" \
  "$REPORT_ROOT/results.json" > "$REPORT_ROOT/artifact_sha256.txt"
echo "[$(date '+%F %T')] CA-1M final107 complete: $REPORT_ROOT/REPORT.md"
