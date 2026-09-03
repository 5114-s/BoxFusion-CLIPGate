#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_scannet_boxer_profile.sh x0_cutr 0,1
#   bash scripts/run_scannet_boxer_profile.sh x0_replay 0,1
#   bash scripts/run_scannet_boxer_profile.sh x1_observer 0,1
#   bash scripts/run_scannet_boxer_profile.sh x2_active 0,1
#   bash scripts/run_scannet_boxer_profile.sh f1_pre_observer 0,1
#   bash scripts/run_scannet_boxer_profile.sh f2_pre_active 0,1
# Score-0.4 paired route (requires x0_cutr_score04 before any replay profile):
#   bash scripts/run_scannet_boxer_profile.sh x0_cutr_score04 0,1
#   bash scripts/run_scannet_boxer_profile.sh x0_replay_score04 0,1
#   bash scripts/run_scannet_boxer_profile.sh x1_observer_score04 0,1
#   bash scripts/run_scannet_boxer_profile.sh x2_active_score04 0,1
PROFILE="${1:-}"
GPU_SPEC="${2:-0}"

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="${BOXFUSION_BOXER_SCENE_LIST:-$CODE_ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
GT_ROOT="${BOXFUSION_SCANNET_GT_ROOT:-$CODE_ROOT/evaluation/data_util/scannet_train_detection_data}"
SCAN_ROOT="${BOXFUSION_SCANNET_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"
ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

case "$PROFILE" in
  x0_cutr)
    CONFIG="$CODE_ROOT/config/scannet_cutr_paired_scorefix.yaml"
    ;;
  x0_replay)
    CONFIG="$CODE_ROOT/config/scannet_cutr_replay_scorefix.yaml"
    ;;
  x1_observer)
    CONFIG="$CODE_ROOT/config/scannet_boxer_observer_scorefix.yaml"
    ;;
  x2_active)
    CONFIG="$CODE_ROOT/config/scannet_boxer_active_scorefix.yaml"
    ;;
  f1_pre_observer)
    CONFIG="$CODE_ROOT/config/scannet_boxer_pre_observer_scorefix.yaml"
    ;;
  f2_pre_active)
    CONFIG="$CODE_ROOT/config/scannet_boxer_pre_active_scorefix.yaml"
    ;;
  x0_cutr_score04)
    CONFIG="$CODE_ROOT/config/scannet_cutr_paired_score04.yaml"
    ;;
  x0_replay_score04)
    CONFIG="$CODE_ROOT/config/scannet_cutr_replay_score04.yaml"
    ;;
  x1_observer_score04)
    CONFIG="$CODE_ROOT/config/scannet_boxer_observer_score04.yaml"
    ;;
  x2_active_score04)
    CONFIG="$CODE_ROOT/config/scannet_boxer_active_score04.yaml"
    ;;
  *)
    echo "Profile must be x0_cutr, x0_replay, x1_observer, x2_active, f1_pre_observer, f2_pre_active, x0_cutr_score04, x0_replay_score04, x1_observer_score04, or x2_active_score04" >&2
    exit 2
    ;;
esac

for required in \
  "$PYTHON" \
  "$SCENE_LIST" \
  "$CONFIG" \
  "$CODE_ROOT/demo.py" \
  "$LIVE_ROOT/models/cutr_rgbd.pth" \
  "$LIVE_ROOT/models/open_clip_pytorch_model.bin" \
  "$GT_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

PRED_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(yaml.safe_load(handle)["data"]["output_dir"])
PY
)"
DIAGNOSTICS_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("boxer", {}).get("diagnostics_dir", ""))
PY
)"
CACHE_MODE="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("mode", "disabled"))
PY
)"
CACHE_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("root", ""))
PY
)"
CACHE_NAMESPACE="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("namespace", ""))
PY
)"
CACHE_BASELINE_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("baseline_prediction_root", ""))
PY
)"
LIST_SHA256="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
LIST_TAG="$(basename "$SCENE_LIST" .txt)-${LIST_SHA256:0:12}"
LOG_ROOT="$CODE_ROOT/logs/boxer_lifting/$PROFILE/$LIST_TAG"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
EVAL_ROOT="$LOG_ROOT/evaluation"
mkdir -p "$PRED_ROOT" "$SCENE_LOG_ROOT" "$EVAL_ROOT"

duplicate_scene="$(
  awk 'NF && $1 !~ /^#/ {count[$1] += 1} END {
    for (scene in count) if (count[scene] > 1) {print scene; exit}
  }' "$SCENE_LIST"
)"
if [[ -n "$duplicate_scene" ]]; then
  echo "Duplicate scene in scene list: $duplicate_scene" >&2
  exit 1
fi
if [[ "$CACHE_MODE" != "disabled" && ( -z "$CACHE_ROOT" || -z "$CACHE_NAMESPACE" ) ]]; then
  echo "Proposal-cache root and namespace are required" >&2
  exit 1
fi
if [[ "$CACHE_MODE" == "replay" && -z "$CACHE_BASELINE_ROOT" ]]; then
  echo "Replay profile requires proposal_cache.baseline_prediction_root" >&2
  exit 1
fi

LOCK_ROOT="$CODE_ROOT/logs/boxer_lifting/locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/${PROFILE}.lock"
if ! flock -n 9; then
  echo "Another process is already writing profile $PROFILE" >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "No GPU was specified" >&2
  exit 1
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index: $gpu" >&2
    exit 1
  fi
done

FINGERPRINT="$(
  {
    sha256sum "$CONFIG"
    # Preserve the fingerprints of completed score-0.5 artifacts.  The only
    # legacy-runner changes below are new profile dispatch entries; score-0.4
    # profiles still bind to the live runner bytes.
    if [[ "$PROFILE" == *_score04 ]]; then
      sha256sum "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh"
    else
      printf '%s  %s\n' \
        "a11efc08572a296e6794afa7547b2a322f3281c625245092bab421c7117adae5" \
        "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh"
    fi
    sha256sum \
      "$CODE_ROOT/demo.py" \
      "$CODE_ROOT/boxfusion/boxer_lifter.py" \
      "$CODE_ROOT/boxfusion/proposal_cache.py" \
      "$CODE_ROOT/boxfusion/cubify_transformer.py" \
      "$CODE_ROOT/boxfusion/instances.py" \
      "$CODE_ROOT/boxfusion/boxes.py" \
      "$CODE_ROOT/boxfusion/box_manager.py" \
      "$CODE_ROOT/boxfusion/box_fusion.py" \
      "$CODE_ROOT/boxfusion/capture_stream.py" \
      "$CODE_ROOT/boxfusion/preprocessor.py" \
      "$CODE_ROOT/tools/utils.py" \
      "$CODE_ROOT/data/pst_1024_0.tiff" \
      "$LIVE_ROOT/models/cutr_rgbd.pth" \
      "$LIVE_ROOT/models/open_clip_pytorch_model.bin"
    printf '%s\n' \
      "live_root=$(readlink -f "$LIVE_ROOT")" \
      "python=$(readlink -f "$PYTHON")" \
      "python_version=$("$PYTHON" --version 2>&1)"
    if [[ "$PROFILE" != "x0_cutr" && "$PROFILE" != "x0_cutr_score04" ]]; then
      git -C "$CODE_ROOT/third_party/boxer" rev-parse HEAD
      sha256sum \
        "$CODE_ROOT/third_party/boxer/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt" \
        "$CODE_ROOT/third_party/boxer/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
    fi
  } | sha256sum | awk '{print $1}'
)"

total="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
worker_count="${#GPUS[@]}"

echo "[$(date '+%F %T')] Starting paired Boxer lifting profile"
echo "[$(date '+%F %T')] profile=$PROFILE, scenes=$total, GPUs=$GPU_SPEC"
echo "[$(date '+%F %T')] config=$CONFIG"
echo "[$(date '+%F %T')] predictions=$PRED_ROOT"
echo "[$(date '+%F %T')] diagnostics=${DIAGNOSTICS_ROOT:-disabled}"
echo "[$(date '+%F %T')] proposal_cache=$CACHE_MODE:${CACHE_NAMESPACE:-disabled}"
echo "[$(date '+%F %T')] fingerprint=$FINGERPRINT"
echo "[$(date '+%F %T')] scene_list_sha256=$LIST_SHA256"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
  "import torch, torchvision, open_clip; assert torch.cuda.is_available(); print('Environment OK:', torch.__version__, torch.version.cuda)"

run_worker() {
  local gpu="$1"
  local shard="$2"
  local shards="$3"
  local index=0
  local completed=0
  local mpl_dir="$LOG_ROOT/mplconfig_gpu${gpu}"
  mkdir -p "$mpl_dir"

  while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" || "$scene" == \#* ]] && continue
    if (( index % shards != shard )); then
      index=$((index + 1))
      continue
    fi

    local pred_path="$PRED_ROOT/${scene}_boxes.pkl"
    local marker_path="$PRED_ROOT/${scene}.run_fingerprint"
    local scene_log="$SCENE_LOG_ROOT/${scene}.log"
    local diagnostic_path=""
    local scene_frames_root
    local scene_input_fingerprint
    local scene_fingerprint
    local cache_manifest=""
    local cache_producer_fingerprint=""
    scene_frames_root="$(
      "$PYTHON" - "$CONFIG" "$scene" <<'PY'
import os
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    configured = yaml.safe_load(handle)["data"]["datadir"]
print(os.path.join(os.path.dirname(os.path.dirname(configured)), sys.argv[2], "frames"))
PY
    )"
    if [[ ! -d "$scene_frames_root" ]]; then
      echo "Missing ScanNet frame directory: $scene_frames_root" >&2
      return 1
    fi
    scene_input_fingerprint="$(
      find "$scene_frames_root" -type f \
        -printf '%P\t%s\t%T@\n' \
        | LC_ALL=C sort \
        | sha256sum \
        | awk '{print $1}'
    )"
    scene_fingerprint="$(
      printf '%s\n%s\n' "$FINGERPRINT" "$scene_input_fingerprint" \
        | sha256sum \
        | awk '{print $1}'
    )"
    if [[ "$CACHE_MODE" == "record" || "$CACHE_MODE" == "replay" ]]; then
      cache_manifest="$CACHE_ROOT/$CACHE_NAMESPACE/$scene/manifest.json"
    fi
    if [[ "$CACHE_MODE" == "replay" ]]; then
      local baseline_marker="$CACHE_BASELINE_ROOT/${scene}.run_fingerprint"
      if [[ ! -s "$baseline_marker" || ! -s "$cache_manifest" ]]; then
        echo "Missing frozen X0 marker/cache for replay: $scene" >&2
        return 1
      fi
      cache_producer_fingerprint="$(tr -d '\n' < "$baseline_marker")"
      local manifest_producer
      manifest_producer="$(
        "$PYTHON" - "$cache_manifest" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["producer_fingerprint"])
PY
      )"
      if [[ "$manifest_producer" != "$cache_producer_fingerprint" ]]; then
        echo "Frozen cache does not match X0 marker: $scene" >&2
        return 1
      fi
      scene_fingerprint="$(
        printf '%s\n%s\n' \
          "$scene_fingerprint" \
          "$(sha256sum "$cache_manifest" | awk '{print $1}')" \
          | sha256sum \
          | awk '{print $1}'
      )"
    elif [[ "$CACHE_MODE" == "record" ]]; then
      cache_producer_fingerprint="$scene_fingerprint"
    fi
    if [[ -n "$DIAGNOSTICS_ROOT" ]]; then
      diagnostic_path="$DIAGNOSTICS_ROOT/${scene}_boxer_lifting.jsonl"
    fi

    if [[ -s "$pred_path" ]]; then
      if [[ ! -s "$marker_path" || "$(tr -d '\n' < "$marker_path")" != "$scene_fingerprint" ]]; then
        echo "Refusing stale/untracked prediction: $pred_path" >&2
        return 1
      fi
      if [[ -n "$diagnostic_path" && ! -s "$diagnostic_path" ]]; then
        echo "Prediction exists but diagnostic is missing: $diagnostic_path" >&2
        return 1
      fi
      if [[ "$CACHE_MODE" == "record" ]]; then
        if [[ ! -s "$cache_manifest" ]]; then
          echo "Prediction exists but proposal cache is missing: $cache_manifest" >&2
          return 1
        fi
        local manifest_producer
        manifest_producer="$(
          "$PYTHON" - "$cache_manifest" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["producer_fingerprint"])
PY
        )"
        if [[ "$manifest_producer" != "$scene_fingerprint" ]]; then
          echo "Prediction/cache fingerprint mismatch: $scene" >&2
          return 1
        fi
      fi
      completed=$((completed + 1))
      echo "[$(date '+%F %T')] [GPU $gpu] $scene already complete"
      index=$((index + 1))
      continue
    fi

    if [[ -n "$diagnostic_path" && -e "$diagnostic_path" ]]; then
      echo "Refusing orphan diagnostic without prediction: $diagnostic_path" >&2
      return 1
    fi
    echo "[$(date '+%F %T')] [GPU $gpu] Running $scene (list index $((index + 1))/$total)"
    (
      cd "$CODE_ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=0 \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT="$cache_producer_fingerprint" \
      BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$cache_producer_fingerprint" \
      OMP_NUM_THREADS=8 \
      MPLCONFIGDIR="$mpl_dir" \
      LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
      "$PYTHON" demo.py scannet \
        --model-path "$LIVE_ROOT/models/cutr_rgbd.pth" \
        --clip_path "$LIVE_ROOT/models/open_clip_pytorch_model.bin" \
        --config "$CONFIG" \
        --device cuda \
        --seq "$scene"
    ) >"$scene_log" 2>&1

    if [[ ! -s "$pred_path" ]]; then
      echo "GPU $gpu did not produce $pred_path" >&2
      return 1
    fi
    if [[ -n "$diagnostic_path" && ! -s "$diagnostic_path" ]]; then
      echo "GPU $gpu did not produce $diagnostic_path" >&2
      return 1
    fi
    if [[ "$CACHE_MODE" == "record" && ! -s "$cache_manifest" ]]; then
      echo "GPU $gpu did not finalize $cache_manifest" >&2
      return 1
    fi
    local marker_temporary="${marker_path}.tmp.$$"
    printf '%s\n' "$scene_fingerprint" >"$marker_temporary"
    mv "$marker_temporary" "$marker_path"
    completed=$((completed + 1))
    local summary
    summary="$(
      grep -E 'Boxer lifting summary|Saving score-preserving predictions' "$scene_log" \
        | tail -n 2 \
        | tr '\n' ' ' \
        || true
    )"
    echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $summary"
    index=$((index + 1))
  done <"$SCENE_LIST"
  echo "[$(date '+%F %T')] [GPU $gpu] Worker completed $completed scenes"
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
  run_worker "${GPUS[$shard]}" "$shard" "$worker_count" &
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
  echo "At least one worker failed; evaluation was not started" >&2
  exit 1
fi

while IFS= read -r scene || [[ -n "$scene" ]]; do
  [[ -z "$scene" || "$scene" == \#* ]] && continue
  if [[ ! -s "$PRED_ROOT/${scene}_boxes.pkl" ]]; then
    echo "Missing requested prediction: $scene" >&2
    exit 1
  fi
done <"$SCENE_LIST"

echo "[$(date '+%F %T')] Inference complete; starting deterministic evaluation"
mkdir -p "$LOG_ROOT/mplconfig_eval"
(
  cd "$CODE_ROOT/evaluation"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$LOG_ROOT/mplconfig_eval" \
  LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
  "$PYTHON" eval_scannet.py \
    --dataset scannet \
    --data_path "$SCAN_ROOT" \
    --gt_root "$GT_ROOT" \
    --dump_dir "$EVAL_ROOT" \
    --scene_list "$SCENE_LIST" \
    --seed 0 \
    --num_point 40000 \
    --cluster_sampling seed_fps \
    --use_3d_nms \
    --use_cls_nms \
    --per_class_proposal \
    --num_workers 0 \
    --gpu 0 \
    --pred_root "$PRED_ROOT"
) >"$LOG_ROOT/eval_stdout.log" 2>&1

grep -E 'eval mAP|eval APrec|eval ARecall' "$LOG_ROOT/eval_stdout.log"
echo "[$(date '+%F %T')] Profile completed: $PROFILE"
