#!/usr/bin/env bash
set -euo pipefail

# Strict fresh-inference live run.
# Usage: bash scripts/run_scannet_f4_stream3dv2_live_full100.sh 0[,1,...]

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 GPU[,GPU,...]" >&2
  exit 2
fi

GPU_SPEC="$1"
ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$ROOT/config/scannet_cbest_f4_stream3dv2_live_score05.yaml"
OFFICIAL_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
SCENE_LIST="${BOXFUSION_LIVE_SCENE_LIST:-$OFFICIAL_SCENE_LIST}"
SKIP_EVAL="${BOXFUSION_LIVE_SKIP_EVAL:-0}"
EXPERIMENT=scannet_cbest_f4_stream3dv2_live_score05
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
EXPECTED_OFFICIAL_LIST_SHA=4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

for required in \
  "$PYTHON" \
  "$CONFIG" \
  "$SCENE_LIST" \
  "$EVAL_RUNNER" \
  "$MODEL" \
  "$CLIP_MODEL"; do
  [[ -e "$required" ]] || {
    echo "Missing required live-run input: $required" >&2
    exit 1
  }
done

case "$SKIP_EVAL" in
  0|1) ;;
  *)
    echo "BOXFUSION_LIVE_SKIP_EVAL must be 0 or 1; got: $SKIP_EVAL" >&2
    exit 2
    ;;
esac

PRED_ROOT=$(awk '$1 == "output_dir:" {print $2; exit}' "$CONFIG")
DIAGNOSTICS_ROOT=$(awk '$1 == "diagnostics_root:" {print $2; exit}' "$CONFIG")
[[ -n "$PRED_ROOT" && "$PRED_ROOT" == "$ROOT/results/"* ]] || {
  echo "Unsafe or missing data.output_dir in $CONFIG: $PRED_ROOT" >&2
  exit 1
}
[[ -n "$DIAGNOSTICS_ROOT" && "$DIAGNOSTICS_ROOT" == "$ROOT/diagnostics/"* ]] || {
  echo "Unsafe or missing online_stream3dv2.diagnostics_root in $CONFIG: $DIAGNOSTICS_ROOT" >&2
  exit 1
}
[[ "$PRED_ROOT" == "$ROOT/results/$EXPERIMENT" ]] || {
  echo "Config output directory does not match the sealed live experiment" >&2
  exit 1
}

LOG_ROOT="$ROOT/logs/$EXPERIMENT"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "No GPU was specified" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || {
    echo "Invalid GPU index: $gpu" >&2
    exit 2
  }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || {
    echo "Duplicate GPU index: $gpu" >&2
    exit 2
  }
  SEEN_GPUS[$gpu]=1
done

if [[ "$SCENE_LIST" == "$OFFICIAL_SCENE_LIST" ]]; then
  [[ "$(sha256sum "$OFFICIAL_SCENE_LIST" | awk '{print $1}')" == \
    "$EXPECTED_OFFICIAL_LIST_SHA" ]] || {
    echo "Official100 scene-list hash mismatch" >&2
    exit 1
  }
fi

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -gt 0 ]] || {
  echo "Scene list is empty: $SCENE_LIST" >&2
  exit 1
}
if [[ "$SCENE_LIST" == "$OFFICIAL_SCENE_LIST" && "${#SCENES[@]}" -ne 100 ]]; then
  echo "Official scene list must contain exactly 100 scenes; found ${#SCENES[@]}" >&2
  exit 1
fi
declare -A SEEN_SCENES=()
for scene in "${SCENES[@]}"; do
  [[ "$scene" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || {
    echo "Invalid ScanNet scene id in $SCENE_LIST: $scene" >&2
    exit 1
  }
  [[ -z "${SEEN_SCENES[$scene]:-}" ]] || {
    echo "Duplicate scene id in $SCENE_LIST: $scene" >&2
    exit 1
  }
  SEEN_SCENES[$scene]=1
  [[ -d "$ROOT/upstream_clean/scannet_readme_frames/$scene/frames" ]] || {
    echo "Missing fresh RGB-D frames for $scene" >&2
    exit 1
  }
done

if [[ "$SKIP_EVAL" -eq 0 && "$SCENE_LIST" != "$OFFICIAL_SCENE_LIST" ]]; then
  echo "A custom BOXFUSION_LIVE_SCENE_LIST requires BOXFUSION_LIVE_SKIP_EVAL=1" >&2
  exit 2
fi

mkdir -p "$PRED_ROOT" "$DIAGNOSTICS_ROOT" "$SCENE_LOG_ROOT" "$MPL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
  echo "Another strict-live driver holds $LOG_ROOT/run.lock" >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
  'import torch, ultralytics, open_clip, yaml; assert torch.cuda.is_available(); print("Strict-live environment OK", torch.__version__, torch.version.cuda, ultralytics.__version__)'

echo "[$(date '+%F %T')] Strict F4 Stream3Dv2 live fresh inference started"
echo "[$(date '+%F %T')] GPUs=$GPU_SPEC workers=${#GPUS[@]} scenes=${#SCENES[@]}"
echo "[$(date '+%F %T')] Config=$CONFIG"
echo "[$(date '+%F %T')] Scene list=$SCENE_LIST"
echo "[$(date '+%F %T')] Predictions=$PRED_ROOT"
echo "[$(date '+%F %T')] Route diagnostics=$DIAGNOSTICS_ROOT"

validate_completed_scene() {
  local scene="$1"
  local pred_path="$PRED_ROOT/${scene}_boxes.pkl"
  local diagnostic_path="$DIAGNOSTICS_ROOT/${scene}.json"
  local scene_log="$SCENE_LOG_ROOT/${scene}.log"

  [[ -s "$pred_path" && -s "$diagnostic_path" && -s "$scene_log" ]] || return 1
  grep -Fq 'Strict live summary |' "$scene_log" || return 1
  ! grep -Eq 'Traceback|Exception in thread|STRICT LIVE FAILURE' "$scene_log" || return 1
  "$PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1], encoding="utf-8")); assert isinstance(p,dict); s=p.get("scene_id"); assert s is None or s==sys.argv[2]' \
    "$diagnostic_path" "$scene" >/dev/null 2>&1 || return 1
}

run_worker() {
  local gpu="$1"
  local shard="$2"
  local shard_count="$3"
  local completed=0
  local index scene pred_path diagnostic_path scene_log

  for index in "${!SCENES[@]}"; do
    (( index % shard_count == shard )) || continue
    scene="${SCENES[$index]}"
    pred_path="$PRED_ROOT/${scene}_boxes.pkl"
    diagnostic_path="$DIAGNOSTICS_ROOT/${scene}.json"
    scene_log="$SCENE_LOG_ROOT/${scene}.log"

    if validate_completed_scene "$scene"; then
      completed=$((completed + 1))
      echo "[$(date '+%F %T')] [GPU $gpu] Reusing complete $scene"
      continue
    fi
    if [[ -e "$pred_path" || -e "$diagnostic_path" ]]; then
      echo "Partial or invalid live artifact exists for $scene; refusing to overwrite" >&2
      return 1
    fi

    echo "[$(date '+%F %T')] [GPU $gpu] Running fresh $scene (list index $((index + 1))/${#SCENES[@]})"
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
      BOXFUSION_STRICT_LIVE=1 \
      "$PYTHON" demo.py scannet \
        --model-path "$MODEL" \
        --clip_path "$CLIP_MODEL" \
        --config "$CONFIG" \
        --device cuda \
        --seq "$scene"
    ) >"$scene_log" 2>&1

    if ! validate_completed_scene "$scene"; then
      echo "Strict live validation failed for $scene; see $scene_log" >&2
      return 1
    fi
    completed=$((completed + 1))
    echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $(grep -F 'Strict live summary |' "$scene_log" | tail -n 1)"
  done
  echo "[$(date '+%F %T')] [GPU $gpu] Worker $shard/$shard_count completed $completed scenes"
}

child_pids=()
cleanup() {
  local pid
  for pid in "${child_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM

for shard in "${!GPUS[@]}"; do
  mkdir -p "$MPL_ROOT/worker${shard}"
  run_worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" &
  child_pids+=("$!")
done

worker_status=0
for pid in "${child_pids[@]}"; do
  if ! wait "$pid"; then
    worker_status=1
  fi
done
trap - INT TERM
if [[ "$worker_status" -ne 0 ]]; then
  echo "At least one strict-live worker failed; evaluation was not started" >&2
  exit 1
fi

for scene in "${SCENES[@]}"; do
  validate_completed_scene "$scene" || {
    echo "Final strict-live artifact validation failed for $scene" >&2
    exit 1
  }
done

echo "[$(date '+%F %T')] All ${#SCENES[@]} requested live scenes passed strict artifact validation"
if [[ "$SKIP_EVAL" -eq 1 ]]; then
  echo "[$(date '+%F %T')] Evaluation skipped by BOXFUSION_LIVE_SKIP_EVAL=1"
  exit 0
fi

prediction_count=$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
diagnostic_count=$(find "$DIAGNOSTICS_ROOT" -maxdepth 1 -type f -name 'scene*.json' | wc -l)
[[ "$prediction_count" -eq 100 ]] || {
  echo "Expected exactly 100 live predictions, found $prediction_count" >&2
  exit 1
}
[[ "$diagnostic_count" -eq 100 ]] || {
  echo "Expected exactly 100 live route diagnostics, found $diagnostic_count" >&2
  exit 1
}

echo "[$(date '+%F %T')] Starting official100 real-score AP evaluation"
bash "$EVAL_RUNNER" "$EXPERIMENT" "$PRED_ROOT"
echo "[$(date '+%F %T')] Strict F4 Stream3Dv2 live official100 complete"
