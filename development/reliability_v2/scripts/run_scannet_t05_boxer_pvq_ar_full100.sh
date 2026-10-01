#!/usr/bin/env bash
set -euo pipefail

# PVQ-AR official100: view-indexed multi-prototype historical CLIP query for
# local rearrangement of ambiguous native 2D-matching edges, on top of the
# Cbest route (Top-K3 + Boxer active + native real-score).
#
# Usage:
#   bash scripts/run_scannet_t05_boxer_pvq_ar_full100.sh shadow 0,1
#   bash scripts/run_scannet_t05_boxer_pvq_ar_full100.sh active  0,1
#
# Controls:
#   BOXFUSION_PVQAR_MAX_SCENES=<1..100>   (partial run skips evaluation)
#   BOXFUSION_PVQAR_SKIP_EVAL=1           (skip the official AP evaluation)

PROFILE="${1:-}"
GPU_SPEC="${2:-0,1}"
MAX_SCENES="${BOXFUSION_PVQAR_MAX_SCENES:-100}"
SKIP_EVAL="${BOXFUSION_PVQAR_SKIP_EVAL:-0}"

ROOT=/data/ZhaoX/BoxFusion
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
CACHE_BASELINE_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/results/boxer_lifting/x0_cutr
CACHE_NAMESPACE=scannet-score05-gap25-postfilter-v2
CACHE_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals
PRODUCER_FINGERPRINT="$(tr -d '\n' < "$CACHE_BASELINE_ROOT/scene0568_00.run_fingerprint")"

case "$PROFILE" in
  shadow)
    CONFIG="${BOXFUSION_PVQAR_CONFIG:-$ROOT/config/scannet_t05_boxer_pvq_ar_shadow_topk3_real_score05.yaml}"
    EXPERIMENT="${BOXFUSION_PVQAR_EXPERIMENT:-scannet_t05_boxer_pvq_ar_shadow_topk3_real_score05}"
    ;;
  active)
    CONFIG="$ROOT/config/scannet_t05_boxer_pvq_ar_active_topk3_real_score05.yaml"
    EXPERIMENT=scannet_t05_boxer_pvq_ar_active_topk3_real_score05
    ;;
  *)
    echo "Profile must be shadow or active" >&2
    exit 2
    ;;
esac

PRED_ROOT="$ROOT/results/$EXPERIMENT"
LOG_ROOT="$ROOT/logs/$EXPERIMENT"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"

[[ "$MAX_SCENES" =~ ^[0-9]+$ ]] || { echo "MAX_SCENES must be an integer" >&2; exit 2; }
(( MAX_SCENES >= 1 && MAX_SCENES <= 100 )) || { echo "MAX_SCENES must be in [1,100]" >&2; exit 2; }
if (( MAX_SCENES < 100 )) && [[ "$SKIP_EVAL" != "1" ]]; then
  echo "A partial run requires BOXFUSION_PVQAR_SKIP_EVAL=1" >&2
  exit 2
fi

for required in "$PYTHON" "$CONFIG" "$SCENE_LIST" "$EVAL_RUNNER" \
  "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" "$ROOT/demo.py" \
  "$ROOT/boxfusion/pvq_ar.py" "$ROOT/boxfusion/instances.py" \
  "$ROOT/boxfusion/box_manager.py"; do
  [[ -e "$required" ]] || { echo "Missing required input: $required" >&2; exit 1; }
done

"$PYTHON" - "$CONFIG" "$PROFILE" "$PRED_ROOT" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
assert cfg["dataset"] == "scannet"
assert cfg["data"]["gap"] == 25
assert cfg["data"]["output_dir"] == sys.argv[3]
assert cfg["detection"]["score_thresh"] == 0.5
assert cfg["lifting"]["backend"] == "boxer"
assert cfg["lifting"]["proposal_cache"]["mode"] == "replay"
assert cfg["lifting"]["proposal_cache"]["namespace"] == "scannet-score05-gap25-postfilter-v2"
assert cfg["lifting"]["boxer"]["mode"] == "active"
assert cfg["lifting"]["boxer"]["apply_stage"] == "post_filter"
assert "boxer_gsa" not in cfg["lifting"]
assert "boxer_mvpr" not in cfg["lifting"]
assert cfg["association"]["appearance_gate"]["enabled"] is False
assert "causal_hungarian" not in cfg["association"]
pvq = cfg["association"]["pvq_ar"]
assert pvq["enabled"] is True
assert pvq["mode"] == sys.argv[2]
assert pvq["max_prototypes"] <= 4
fusion = cfg["box_fusion"]
assert fusion["use"] is True
assert fusion["reliable_views"]["enabled"] is True
assert fusion["reliable_views"]["top_k"] == 3
assert fusion["reliable_views"]["min_views"] == 3
assert not fusion.get("capf", {}).get("enabled", False)
assert not fusion.get("vapf_lite", {}).get("enabled", False)
assert not fusion.get("maskdepth_pfo", {}).get("enabled", False)
assert not cfg.get("online_stream3dv2", {}).get("enabled", False)
assert not cfg.get("online_stream3dv3", {}).get("enabled", False)
assert cfg["eval"] is True
assert cfg["vis"]["rerun"] is False
print(f"PVQ-AR {sys.argv[2]} official100 protocol semantics OK")
PY

mapfile -t ALL_SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#ALL_SCENES[@]}" -eq 100 ]] || { echo "Scene list must contain exactly 100 scenes" >&2; exit 1; }
SCENES=("${ALL_SCENES[@]:0:MAX_SCENES}")
for scene in "${SCENES[@]}"; do
  [[ "$scene" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || { echo "Invalid scene id: $scene" >&2; exit 1; }
  [[ -d "$ROOT/upstream_clean/scannet_readme_frames/$scene/frames" ]] || {
    echo "Missing RGB-D frames: $scene" >&2; exit 1; }
  [[ -s "$CACHE_ROOT/$CACHE_NAMESPACE/$scene/manifest.json" ]] || {
    echo "Missing proposal-cache manifest: $scene" >&2; exit 1; }
done

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -ge 1 ]] || { echo "No GPU specified" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo "Duplicate GPU: $gpu" >&2; exit 2; }
  SEEN_GPUS[$gpu]=1
done

mkdir -p "$PRED_ROOT" "$SCENE_LOG_ROOT" "$MPL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
  echo "Another PVQ-AR driver holds $LOG_ROOT/run.lock" >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c 'import torch,open_clip,scipy; assert torch.cuda.is_available(); print("PVQ-AR environment OK", torch.__version__)'

echo "[$(date '+%F %T')] PVQ-AR $PROFILE official100 started"
echo "[$(date '+%F %T')] GPUs=$GPU_SPEC scenes=$MAX_SCENES config=$CONFIG"
echo "[$(date '+%F %T')] predictions=$PRED_ROOT"

validate_completed_scene() {
  local scene="$1"
  local pred="$PRED_ROOT/${scene}_boxes.pkl"
  local log="$SCENE_LOG_ROOT/${scene}.log"
  [[ -s "$pred" && -s "$log" ]] || return 1
  [[ "$(grep -Fc 'PVQ-AR summary |' "$log")" -eq 1 ]] || return 1
  [[ "$(grep -Fc 'Saving score-preserving predictions:' "$log")" -eq 1 ]] || return 1
  [[ "$(grep -Ec 'Average FPS: [0-9.]+' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|PVQ-AR FAILURE' "$log" || return 1
}

run_worker() {
  local gpu="$1" shard="$2" shard_count="$3"
  local index scene pred log
  for index in "${!SCENES[@]}"; do
    (( index % shard_count == shard )) || continue
    scene="${SCENES[$index]}"
    pred="$PRED_ROOT/${scene}_boxes.pkl"
    log="$SCENE_LOG_ROOT/${scene}.log"
    if validate_completed_scene "$scene"; then
      echo "[$(date '+%F %T')] [GPU $gpu] Reusing complete $scene"
      continue
    fi
    if [[ -e "$pred" ]]; then
      echo "Stale prediction exists for $scene; refusing to overwrite" >&2
      return 1
    fi
    echo "[$(date '+%F %T')] [GPU $gpu] Running $scene (list index $((index + 1))/$MAX_SCENES)"
    (
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=0 \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      HF_HUB_OFFLINE=1 \
      BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$PRODUCER_FINGERPRINT" \
      OMP_NUM_THREADS=8 \
      MPLCONFIGDIR="$MPL_ROOT/worker${shard}" \
      LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
      "$PYTHON" demo.py scannet \
        --model-path "$MODEL" \
        --clip_path "$CLIP_MODEL" \
        --class_txt "$CLASS_TXT" \
        --config "$CONFIG" \
        --device cuda \
        --seq "$scene"
    ) >"$log" 2>&1 || { echo "Scene failed: $scene (see $log)" >&2; return 1; }
    if ! validate_completed_scene "$scene"; then
      echo "PVQ-AR artifact validation failed: $scene" >&2
      return 1
    fi
    echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $(grep -F 'PVQ-AR summary |' "$log" | tail -n 1)"
  done
  echo "[$(date '+%F %T')] [GPU $gpu] Worker $shard/$shard_count done"
}

child_pids=()
cleanup() {
  local pid
  for pid in "${child_pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup INT TERM
mkdir -p "$MPL_ROOT"/worker{0,1}
for shard in "${!GPUS[@]}"; do
  run_worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" &
  child_pids+=("$!")
done
worker_status=0
for pid in "${child_pids[@]}"; do
  wait "$pid" || worker_status=1
done
trap - INT TERM
[[ "$worker_status" -eq 0 ]] || { echo "A PVQ-AR worker failed; evaluation not started" >&2; exit 1; }

for scene in "${SCENES[@]}"; do
  validate_completed_scene "$scene" || { echo "Final validation failed: $scene" >&2; exit 1; }
done
echo "[$(date '+%F %T')] All $MAX_SCENES PVQ-AR $PROFILE scenes passed validation"
if (( MAX_SCENES < 100 )) || [[ "$SKIP_EVAL" == "1" ]]; then
  echo "[$(date '+%F %T')] Evaluation skipped"
  exit 0
fi
bash "$EVAL_RUNNER" "$EXPERIMENT" "$PRED_ROOT"
echo "[$(date '+%F %T')] PVQ-AR $PROFILE official100 complete"
