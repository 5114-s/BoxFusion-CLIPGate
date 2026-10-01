#!/usr/bin/env bash
set -euo pipefail

# Boxer + Reliable-view Top-K3 + CAPF-online + native real-score official100.
# Usage: bash scripts/run_scannet_t05_boxer_capf_online_topk3_real_score_full100.sh [GPU[,GPU...]]
# One worker is assigned to each distinct GPU.

GPU_SPEC="${1:-0,1}"
MAX_SCENES="${BOXFUSION_CAPF_MAX_SCENES:-100}"
SKIP_EVAL="${BOXFUSION_CAPF_SKIP_EVAL:-0}"
PREFLIGHT_ONLY="${BOXFUSION_CAPF_PREFLIGHT_ONLY:-0}"

ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion2
PYTHON="$ENV_ROOT/bin/python"
CONFIG="${BOXFUSION_CAPF_CONFIG:-$ROOT/config/scannet_t05_boxer_capf_topk3_real_score05.yaml}"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
EXPERIMENT="${BOXFUSION_CAPF_EXPERIMENT:-scannet_t05_boxer_capf_topk3_real_score05}"
PRED_ROOT="$ROOT/results/$EXPERIMENT"
BOXER_DIAGNOSTICS="${BOXFUSION_CAPF_BOXER_DIAGNOSTICS:-$ROOT/diagnostics/t05_boxer_capf/boxer_lifting}"
LOG_ROOT="$ROOT/logs/$EXPERIMENT"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"
RUNNER="$ROOT/scripts/run_scannet_t05_boxer_capf_online_topk3_real_score_full100.sh"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$ROOT/data/class_features.pt"
PST="$ROOT/data/pst_1024_0.tiff"
BOXER_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer
BOXER_CHECKPOINT="$BOXER_ROOT/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CHECKPOINT="$BOXER_ROOT/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
CACHE_BASELINE_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/results/boxer_lifting/x0_cutr
CACHE_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals
CACHE_NAMESPACE=scannet-score05-gap25-postfilter-v2
EXPECTED_CACHE_FINGERPRINT=ba44e29386d2c2f76bb927e00f02b62cfc5ee4f188a94408c32ce91757f4462d
EXPECTED_SCENE_LIST_SHA=4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5
EXPECTED_BOXER_COMMIT=1f86542dc342a4b1d474c87c97c5d1d6566d9148
EXPECTED_BOXER_SHA=d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f
EXPECTED_DINO_SHA=4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea
EXPECTED_MODEL_SHA=856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217
EXPECTED_CLIP_SHA=9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4
EXPECTED_CLASS_TXT_SHA=0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9
EXPECTED_CLASS_FEATURES_SHA=49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197
EXPECTED_PST_SHA=867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

for switch in "$SKIP_EVAL" "$PREFLIGHT_ONLY"; do
  case "$switch" in
    0|1) ;;
    *)
      echo "BOXFUSION_CAPF_SKIP_EVAL and BOXFUSION_CAPF_PREFLIGHT_ONLY must be 0 or 1" >&2
      exit 2
      ;;
  esac
done
[[ "$MAX_SCENES" =~ ^[0-9]+$ ]] || {
  echo "BOXFUSION_CAPF_MAX_SCENES must be an integer" >&2
  exit 2
}
(( MAX_SCENES >= 1 && MAX_SCENES <= 100 )) || {
  echo "BOXFUSION_CAPF_MAX_SCENES must be in [1,100]" >&2
  exit 2
}
if (( MAX_SCENES < 100 )) && [[ "$SKIP_EVAL" -ne 1 ]]; then
  echo "A partial CAPF run requires BOXFUSION_CAPF_SKIP_EVAL=1" >&2
  exit 2
fi

mapfile -t BOXFUSION_SOURCES < <(
  rg --files "$ROOT/boxfusion" | rg '\.py$' | sort
)
FINGERPRINT_SOURCES=(
  "$ROOT/demo.py"
  "$ROOT/tools/utils.py"
  "${BOXFUSION_SOURCES[@]}"
)
for required in \
  "$PYTHON" "$CONFIG" "$SCENE_LIST" "$RUNNER" "$EVAL_RUNNER" \
  "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" "$CLASS_FEATURES" "$PST" \
  "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT" "${FINGERPRINT_SOURCES[@]}"; do
  [[ -e "$required" ]] || {
    echo "Missing required CAPF input: $required" >&2
    exit 1
  }
done

[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == "$EXPECTED_SCENE_LIST_SHA" ]] || {
  echo "Official100 scene-list hash mismatch" >&2
  exit 1
}
[[ "$(git -C "$BOXER_ROOT" rev-parse HEAD)" == "$EXPECTED_BOXER_COMMIT" ]] || {
  echo "Boxer commit mismatch" >&2
  exit 1
}
for asset_and_hash in \
  "$BOXER_CHECKPOINT:$EXPECTED_BOXER_SHA" \
  "$DINO_CHECKPOINT:$EXPECTED_DINO_SHA" \
  "$MODEL:$EXPECTED_MODEL_SHA" \
  "$CLIP_MODEL:$EXPECTED_CLIP_SHA" \
  "$CLASS_TXT:$EXPECTED_CLASS_TXT_SHA" \
  "$CLASS_FEATURES:$EXPECTED_CLASS_FEATURES_SHA" \
  "$PST:$EXPECTED_PST_SHA"; do
  asset="${asset_and_hash%:*}"
  expected="${asset_and_hash##*:}"
  [[ "$(sha256sum "$asset" | awk '{print $1}')" == "$expected" ]] || {
    echo "Asset hash mismatch: $asset" >&2
    exit 1
  }
done

mapfile -t ALL_SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#ALL_SCENES[@]}" -eq 100 ]] || {
  echo "Official scene list must contain exactly 100 scenes" >&2
  exit 1
}
SCENES=("${ALL_SCENES[@]:0:MAX_SCENES}")
for scene in "${ALL_SCENES[@]}"; do
  [[ "$scene" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || {
    echo "Invalid ScanNet scene id: $scene" >&2
    exit 1
  }
  [[ -d "$ROOT/upstream_clean/scannet_readme_frames/$scene/frames" ]] || {
    echo "Missing RGB-D frames: $scene" >&2
    exit 1
  }
  [[ -s "$CACHE_BASELINE_ROOT/${scene}.run_fingerprint" ]] || {
    echo "Missing proposal-cache producer marker: $scene" >&2
    exit 1
  }
  [[ -s "$CACHE_ROOT/$CACHE_NAMESPACE/$scene/manifest.json" ]] || {
    echo "Missing proposal-cache manifest: $scene" >&2
    exit 1
  }
  [[ "$(tr -d '\n' < "$CACHE_BASELINE_ROOT/${scene}.run_fingerprint")" == "$EXPECTED_CACHE_FINGERPRINT" ]] || {
    echo "Unexpected proposal-cache producer fingerprint: $scene" >&2
    exit 1
  }
done

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "No GPU specified" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || {
    echo "Duplicate GPU index is not allowed: $gpu" >&2
    exit 2
  }
  SEEN_GPUS[$gpu]=1
done

"$PYTHON" - "$CONFIG" "$PRED_ROOT" "$BOXER_DIAGNOSTICS" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
assert cfg["dataset"] == "scannet"
assert cfg["data"]["gap"] == 25
assert cfg["data"]["output_dir"] == sys.argv[2]
assert cfg["detection"]["score_thresh"] == 0.5
assert cfg["lifting"]["backend"] == "boxer"
assert cfg["lifting"]["proposal_cache"]["mode"] == "replay"
assert cfg["lifting"]["proposal_cache"]["namespace"] == "scannet-score05-gap25-postfilter-v2"
assert cfg["lifting"]["proposal_cache"]["expected_fingerprint"] == "ba44e29386d2c2f76bb927e00f02b62cfc5ee4f188a94408c32ce91757f4462d"
assert cfg["lifting"]["boxer"]["mode"] == "active"
assert cfg["lifting"]["boxer"]["apply_stage"] == "post_filter"
assert cfg["lifting"]["boxer"]["diagnostics_dir"] == sys.argv[3]
assert "boxer_gsa" not in cfg["lifting"]
assert "boxer_mvpr" not in cfg["lifting"]
assert cfg["association"]["appearance_gate"]["enabled"] is False
assert "causal_hungarian" not in cfg["association"]
fusion = cfg["box_fusion"]
assert fusion["use"] is True
assert fusion["reliable_views"]["enabled"] is True
assert fusion["reliable_views"]["top_k"] == 3
assert fusion["reliable_views"]["min_views"] == 3
assert fusion["capf"]["enabled"] is True
assert fusion["capf"]["min_views"] == 3
assert not fusion.get("vapf_lite", {}).get("enabled", False)
assert not fusion.get("maskdepth_pfo", {}).get("enabled", False)
assert not cfg.get("online_stream3dv2", {}).get("enabled", False)
assert not cfg.get("online_stream3dv3", {}).get("enabled", False)
assert cfg["eval"] is True
assert cfg["vis"]["rerun"] is False
print("CAPF-online official100 protocol semantics OK")
PY

"$PYTHON" - "${FINGERPRINT_SOURCES[@]}" <<'PY'
from pathlib import Path
import sys

for value in sys.argv[1:]:
    path = Path(value)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print(f"Python syntax OK: {len(sys.argv) - 1} sources")
PY
bash -n "$RUNNER"

compute_fingerprint() {
  {
    sha256sum \
      "$CONFIG" "$SCENE_LIST" "$RUNNER" "$EVAL_RUNNER" \
      "${FINGERPRINT_SOURCES[@]}" \
      "$PST" "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" "$CLASS_FEATURES" \
      "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT"
    printf '%s\n' \
      "python=$(readlink -f "$PYTHON")" \
      "python_version=$($PYTHON --version 2>&1)" \
      "boxer_commit=$EXPECTED_BOXER_COMMIT"
  } | sha256sum | awk '{print $1}'
}

FINGERPRINT="$(compute_fingerprint)"
if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  echo "CAPF-online preflight OK"
  echo "Scenes=$MAX_SCENES GPUs=$GPU_SPEC fingerprint=$FINGERPRINT"
  exit 0
fi

mkdir -p "$PRED_ROOT" "$BOXER_DIAGNOSTICS" "$SCENE_LOG_ROOT" "$MPL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
  echo "Another CAPF-online driver holds $LOG_ROOT/run.lock" >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
  'import torch,open_clip,scipy,shapely; assert torch.cuda.is_available(); print("CAPF environment OK",torch.__version__,torch.version.cuda)'

echo "[$(date '+%F %T')] CAPF-online inference started"
echo "[$(date '+%F %T')] GPUs=$GPU_SPEC workers=${#GPUS[@]} scenes=$MAX_SCENES fingerprint=$FINGERPRINT"

validate_completed_scene() {
  local scene="$1"
  local pred="$PRED_ROOT/${scene}_boxes.pkl"
  local marker="$PRED_ROOT/${scene}.run_fingerprint"
  local diagnostic="$BOXER_DIAGNOSTICS/${scene}_boxer_lifting.jsonl"
  local log="$SCENE_LOG_ROOT/${scene}.log"
  [[ -s "$pred" && -s "$marker" && -s "$diagnostic" && -s "$log" ]] || return 1
  [[ "$(tr -d '\n' < "$marker")" == "$FINGERPRINT" ]] || return 1
  [[ "$(grep -Fc 'CAPF summary |' "$log")" -eq 1 ]] || return 1
  [[ "$(grep -Fc 'Reliable-view fusion summary |' "$log")" -eq 1 ]] || return 1
  [[ "$(grep -Fc 'Boxer lifting summary |' "$log")" -eq 1 ]] || return 1
  [[ "$(grep -Fc 'Saving score-preserving predictions:' "$log")" -eq 1 ]] || return 1
  [[ "$(grep -Ec 'Cost: [0-9.]+ s Average FPS: [0-9.]+' "$log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|Exception in thread|CAPF FAILURE' "$log" || return 1
  "$PYTHON" - "$pred" <<'PY' >/dev/null 2>&1
import pickle
import sys

import numpy as np

with open(sys.argv[1], "rb") as handle:
    payload = pickle.load(handle)
assert isinstance(payload, (list, tuple)) and len(payload) == 1
rows = payload[0]
assert isinstance(rows, (list, tuple))
scores = []
for row in rows:
    assert isinstance(row, tuple) and len(row) == 3
    class_id, corners, score = row
    assert int(class_id) == 0
    corners = np.asarray(corners)
    assert corners.shape == (8, 3) and np.isfinite(corners).all()
    score = float(score)
    assert np.isfinite(score) and 0.0 <= score <= 1.0
    scores.append(score)
if scores:
    assert any(score < 0.999999 for score in scores)
PY
}

run_worker() {
  local gpu="$1"
  local shard="$2"
  local shard_count="$3"
  local completed=0
  local index scene pred marker diagnostic log producer

  for index in "${!SCENES[@]}"; do
    (( index % shard_count == shard )) || continue
    scene="${SCENES[$index]}"
    pred="$PRED_ROOT/${scene}_boxes.pkl"
    marker="$PRED_ROOT/${scene}.run_fingerprint"
    diagnostic="$BOXER_DIAGNOSTICS/${scene}_boxer_lifting.jsonl"
    log="$SCENE_LOG_ROOT/${scene}.log"

    [[ "$(compute_fingerprint)" == "$FINGERPRINT" ]] || {
      echo "CAPF source/asset fingerprint changed before $scene; refusing a mixed run" >&2
      return 1
    }
    if validate_completed_scene "$scene"; then
      completed=$((completed + 1))
      echo "[$(date '+%F %T')] [GPU $gpu] Reusing complete $scene"
      continue
    fi
    if [[ -e "$pred" || -e "$marker" || -e "$diagnostic" || -e "$log" ]]; then
      echo "Partial, stale, or invalid CAPF artifact exists for $scene; refusing to overwrite" >&2
      return 1
    fi
    producer="$(tr -d '\n' < "$CACHE_BASELINE_ROOT/${scene}.run_fingerprint")"
    [[ "$producer" =~ ^[0-9a-f]{64}$ ]] || {
      echo "Invalid proposal-cache producer marker: $scene" >&2
      return 1
    }

    echo "[$(date '+%F %T')] [GPU $gpu] Running $scene (list index $((index + 1))/100)"
    (
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=0 \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      HF_HUB_OFFLINE=1 \
      BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$producer" \
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
    ) >"$log" 2>&1

    [[ "$(compute_fingerprint)" == "$FINGERPRINT" ]] || {
      echo "CAPF source/asset fingerprint changed while running $scene" >&2
      return 1
    }
    [[ -s "$pred" && -s "$diagnostic" ]] || {
      echo "CAPF scene failed to produce outputs: $scene; see $log" >&2
      return 1
    }
    printf '%s\n' "$FINGERPRINT" >"${marker}.tmp.$$"
    mv "${marker}.tmp.$$" "$marker"
    if ! validate_completed_scene "$scene"; then
      echo "CAPF artifact validation failed: $scene; see $log" >&2
      return 1
    fi
    completed=$((completed + 1))
    echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $(grep -F 'CAPF summary |' "$log" | tail -n 1)"
  done
  echo "[$(date '+%F %T')] [GPU $gpu] Worker $shard/$shard_count completed=$completed"
}

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

worker_status=0
for pid in "${child_pids[@]}"; do
  wait "$pid" || worker_status=1
done
trap - INT TERM
[[ "$worker_status" -eq 0 ]] || {
  echo "At least one CAPF worker failed; evaluation was not started" >&2
  exit 1
}

for scene in "${SCENES[@]}"; do
  validate_completed_scene "$scene" || {
    echo "Final CAPF artifact validation failed: $scene" >&2
    exit 1
  }
done
[[ "$(compute_fingerprint)" == "$FINGERPRINT" ]] || {
  echo "CAPF source/asset fingerprint changed before evaluation" >&2
  exit 1
}

echo "[$(date '+%F %T')] All $MAX_SCENES requested CAPF scenes passed validation"
if (( MAX_SCENES < 100 )) || [[ "$SKIP_EVAL" -eq 1 ]]; then
  echo "[$(date '+%F %T')] Evaluation skipped"
  exit 0
fi

prediction_count="$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)"
marker_count="$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*.run_fingerprint' | wc -l)"
diagnostic_count="$(find "$BOXER_DIAGNOSTICS" -maxdepth 1 -type f -name 'scene*_boxer_lifting.jsonl' | wc -l)"
[[ "$prediction_count" -eq 100 ]] || {
  echo "Expected exactly 100 CAPF predictions, found $prediction_count" >&2
  exit 1
}
[[ "$marker_count" -eq 100 ]] || {
  echo "Expected exactly 100 CAPF markers, found $marker_count" >&2
  exit 1
}
[[ "$diagnostic_count" -eq 100 ]] || {
  echo "Expected exactly 100 Boxer diagnostics, found $diagnostic_count" >&2
  exit 1
}

echo "[$(date '+%F %T')] Starting official100 native real-score AP evaluation"
bash "$EVAL_RUNNER" "$EXPERIMENT" "$PRED_ROOT"
echo "[$(date '+%F %T')] CAPF-online official100 complete"
