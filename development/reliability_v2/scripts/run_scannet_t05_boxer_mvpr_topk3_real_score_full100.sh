#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,1}"
ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion2
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$ROOT/config/scannet_t05_boxer_mvpr_topk3_real_score05.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
EXPERIMENT=scannet_t05_boxer_mvpr_topk3_real_score05
PRED_ROOT="$ROOT/results/$EXPERIMENT"
MVPR_DIAGNOSTICS="$ROOT/diagnostics/t05_boxer_mvpr/mvpr"
BOXER_DIAGNOSTICS="$ROOT/diagnostics/t05_boxer_mvpr/boxer_lifting"
LOG_ROOT="$ROOT/logs/$EXPERIMENT"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
EXPECTED_LIST_SHA=4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

for required in "$PYTHON" "$CONFIG" "$SCENE_LIST" "$EVAL_RUNNER" "$MODEL" "$CLIP_MODEL"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == "$EXPECTED_LIST_SHA" ]] || {
  echo "Official100 scene-list hash mismatch" >&2
  exit 1
}

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]] || {
  echo "Official scene list must contain exactly 100 scenes" >&2
  exit 1
}

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "No GPU specified" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }
  SEEN_GPUS[$gpu]=1
done

mkdir -p "$PRED_ROOT" "$MVPR_DIAGNOSTICS" "$BOXER_DIAGNOSTICS" "$SCENE_LOG_ROOT" "$MPL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Another Boxer-MVPR full100 driver is active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

"$PYTHON" - "$CONFIG" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
actual = {
    "score": cfg["detection"]["score_thresh"],
    "boxer": cfg["lifting"]["boxer"]["mode"],
    "mvpr": cfg["lifting"]["boxer_mvpr"]["enabled"],
    "low": cfg["lifting"]["boxer_mvpr"]["low_score_min"],
    "views": cfg["lifting"]["boxer_mvpr"]["minimum_views"],
    "topk": cfg["box_fusion"]["reliable_views"]["top_k"],
    "appearance": cfg["association"]["appearance_gate"]["enabled"],
}
expected = {
    "score": 0.5, "boxer": "active", "mvpr": True, "low": 0.4,
    "views": 3, "topk": 3, "appearance": False,
}
assert actual == expected, (actual, expected)
print("Protocol OK:", actual)
PY

validate_scene() {
  local scene="$1"
  local pred="$PRED_ROOT/${scene}_boxes.pkl"
  local diag="$MVPR_DIAGNOSTICS/${scene}.json"
  local log="$SCENE_LOG_ROOT/${scene}.log"
  [[ -s "$pred" && -s "$diag" && -s "$log" ]] || return 1
  grep -Fq 'Boxer-MVPR summary:' "$log" || return 1
  grep -Fq 'Saving score-preserving predictions:' "$log" || return 1
  ! grep -Eq 'Traceback|Exception in thread' "$log" || return 1
  "$PYTHON" - "$pred" "$diag" "$scene" <<'PY' >/dev/null 2>&1
import json, pickle, sys
payload = pickle.load(open(sys.argv[1], "rb"))
assert isinstance(payload, (list, tuple)) and len(payload) == 1
diag = json.load(open(sys.argv[2], encoding="utf-8"))
assert diag["schema"] == "boxfusion.boxer_mvpr.v1"
assert diag["scene_id"] == sys.argv[3]
assert diag["stats"]["keyframes"] > 0
PY
}

run_worker() {
  local gpu="$1"
  local shard="$2"
  local shard_count="$3"
  local index scene log
  for index in "${!SCENES[@]}"; do
    (( index % shard_count == shard )) || continue
    scene="${SCENES[$index]}"
    log="$SCENE_LOG_ROOT/${scene}.log"
    if validate_scene "$scene"; then
      echo "[$(date '+%F %T')] [GPU $gpu] Reusing $scene"
      continue
    fi
    [[ ! -e "$PRED_ROOT/${scene}_boxes.pkl" ]] || {
      echo "Invalid existing prediction blocks resume: $scene" >&2
      return 1
    }
    echo "[$(date '+%F %T')] [GPU $gpu] Running $scene ($((index + 1))/100)"
    (
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=0 \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      MPLCONFIGDIR="$MPL_ROOT/worker${shard}" \
      LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
      "$PYTHON" demo.py scannet \
        --model-path "$MODEL" \
        --clip_path "$CLIP_MODEL" \
        --config "$CONFIG" \
        --device cuda \
        --seq "$scene"
    ) >"$log" 2>&1
    validate_scene "$scene" || {
      echo "Scene validation failed: $scene; see $log" >&2
      return 1
    }
    echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $(grep -F 'Boxer-MVPR summary:' "$log" | tail -n 1)"
  done
}

echo "[$(date '+%F %T')] Boxer-MVPR Top-K3 real-score official100 started on GPUs=$GPU_SPEC"
child_pids=()
cleanup() {
  local pid
  for pid in "${child_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM
for shard in "${!GPUS[@]}"; do
  mkdir -p "$MPL_ROOT/worker${shard}"
  run_worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" &
  child_pids+=("$!")
done

status=0
for pid in "${child_pids[@]}"; do
  wait "$pid" || status=1
done
trap - INT TERM
[[ "$status" -eq 0 ]] || { echo "At least one worker failed" >&2; exit 1; }

for scene in "${SCENES[@]}"; do
  validate_scene "$scene" || { echo "Final validation failed: $scene" >&2; exit 1; }
done
[[ "$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)" -eq 100 ]] || {
  echo "Prediction root does not contain exactly 100 scenes" >&2
  exit 1
}

echo "[$(date '+%F %T')] Starting official100 real-score AP evaluation"
bash "$EVAL_RUNNER" "$EXPERIMENT" "$PRED_ROOT"
echo "[$(date '+%F %T')] Boxer-MVPR official100 complete"
