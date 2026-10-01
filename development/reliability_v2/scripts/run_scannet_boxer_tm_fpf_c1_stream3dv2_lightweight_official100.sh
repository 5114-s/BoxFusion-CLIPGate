#!/usr/bin/env bash
set -euo pipefail

# Fresh official100 run: exactly 100 scenes, sharded 50/50 over GPU 0 and 1.
# Usage: bash scripts/run_scannet_boxer_tm_fpf_c1_stream3dv2_lightweight_official100.sh 0,1

if [[ "$#" -ne 1 || "$1" != "0,1" ]]; then
  echo "Usage: $0 0,1 (official100 requires GPU 0 and 1)" >&2
  exit 2
fi

GPU_SPEC="$1"
ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$ROOT/config/scannet_boxer_tm_fpf_c1_stream3dv2_lightweight_score05.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
RUNNER="$ROOT/scripts/run_scannet_boxer_tm_fpf_c1_stream3dv2_lightweight_official100.sh"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
SUMMARY_TOOL="$ROOT/tools/summarize_stream3dv2_live_official100.py"
EXPERIMENT=scannet_boxer_tm_fpf_c1_stream3dv2_lightweight_score05
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
FASTSAM_CHECKPOINT=/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/checkpoints/FastSAM.pt
BOXER_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer
BOXER_CHECKPOINT="$BOXER_ROOT/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CHECKPOINT="$BOXER_ROOT/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
EXPECTED_OFFICIAL_LIST_SHA=4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5
EXPECTED_BOXER_COMMIT=1f86542dc342a4b1d474c87c97c5d1d6566d9148
PREFLIGHT_ONLY="${BOXFUSION_LIGHTWEIGHT_PREFLIGHT_ONLY:-0}"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

case "$PREFLIGHT_ONLY" in
  0|1) ;;
  *)
    echo "BOXFUSION_LIGHTWEIGHT_PREFLIGHT_ONLY must be 0 or 1" >&2
    exit 2
    ;;
esac

mapfile -t BOXFUSION_PY_SOURCES < <(
  rg --files "$ROOT/boxfusion" | rg '\.py$' | sort
)
FINGERPRINT_SOURCES=(
  "$CONFIG"
  "$RUNNER"
  "$ROOT/demo.py"
  "$ROOT/tools/utils.py"
  "$SUMMARY_TOOL"
  "$EVAL_RUNNER"
  "${BOXFUSION_PY_SOURCES[@]}"
)

for required in \
  "$PYTHON" "$CONFIG" "$SCENE_LIST" "$RUNNER" "$EVAL_RUNNER" \
  "$SUMMARY_TOOL" "$MODEL" "$CLIP_MODEL" "$FASTSAM_CHECKPOINT" \
  "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT" "${FINGERPRINT_SOURCES[@]}"; do
  [[ -e "$required" ]] || {
    echo "Missing required official100 input: $required" >&2
    exit 1
  }
done

[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  "$EXPECTED_OFFICIAL_LIST_SHA" ]] || {
  echo "Official100 scene-list hash mismatch" >&2
  exit 1
}
[[ "$(git -C "$BOXER_ROOT" rev-parse HEAD)" == "$EXPECTED_BOXER_COMMIT" ]] || {
  echo "Boxer commit mismatch" >&2
  exit 1
}

"$PYTHON" - "$CONFIG" "$ROOT" "$EXPERIMENT" <<'PY'
from pathlib import Path
import sys
import yaml

config_path, root, experiment = sys.argv[1:]
with open(config_path, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

assert cfg["dataset"] == "scannet"
assert int(cfg["data"]["gap"]) == 25
assert cfg["data"]["output_dir"] == str(Path(root) / "results" / experiment)
assert float(cfg["detection"]["score_thresh"]) == 0.5
assert cfg["lifting"]["backend"] == "boxer"
boxer = cfg["lifting"]["boxer"]
assert boxer["mode"] == "active" and boxer["apply_stage"] == "post_filter"
assert boxer["cache_image_features"] is True
assert cfg["association"]["appearance_gate"]["enabled"] is False
fusion = cfg["box_fusion"]
assert fusion["use"] is True
assert fusion["reliable_views"]["enabled"] is True
assert int(fusion["reliable_views"]["top_k"]) == 3
assert fusion["capf"]["enabled"] is False
assert fusion["vapf_lite"]["enabled"] is False
tm = fusion["tm_fpf_c1"]
assert tm["enabled"] is True
assert int(tm["max_accepted_faces"]) == 1
assert int(tm["maximum_views"]) == 5
route = cfg["online_stream3dv2"]
assert route["enabled"] is True
light = route["lightweight"]
assert light["enabled"] is True
assert light["depth_trigger"]["enabled"] is True
assert int(light["fastsam_top_k"]["box_shortlist"]) == 12
assert int(light["fastsam_top_k"]["mask_cap"]) == 8
assert light["conditional_f2"] is True
assert int(light["f4_top_m_tracks"]) == 8
assert light["terminal_clip"]["enabled"] is True
assert int(light["terminal_clip"]["batch_size"]) == 32
assert route["sam3"]["enabled"] is False
assert cfg["online_stream3dv3"]["enabled"] is False
assert cfg["eval"] is True and cfg["vis"]["rerun"] is False
print("Lightweight Boxer + TM-FPF-C1 configuration semantics OK")
PY

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]] || {
  echo "Official scene list must contain exactly 100 scenes; found ${#SCENES[@]}" >&2
  exit 1
}
declare -A SEEN_SCENES=()
shard_zero=0
shard_one=0
for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"
  [[ "$scene" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || {
    echo "Invalid ScanNet scene id: $scene" >&2
    exit 1
  }
  [[ -z "${SEEN_SCENES[$scene]:-}" ]] || {
    echo "Duplicate scene id: $scene" >&2
    exit 1
  }
  SEEN_SCENES[$scene]=1
  [[ -d "$ROOT/upstream_clean/scannet_readme_frames/$scene/frames" ]] || {
    echo "Missing fresh RGB-D frames for $scene" >&2
    exit 1
  }
  if (( index % 2 == 0 )); then
    shard_zero=$((shard_zero + 1))
  else
    shard_one=$((shard_one + 1))
  fi
done
[[ "$shard_zero" -eq 50 && "$shard_one" -eq 50 ]] || {
  echo "Official100 shard error: GPU0=$shard_zero GPU1=$shard_one" >&2
  exit 1
}

PRED_ROOT=$(awk '$1 == "output_dir:" {print $2; exit}' "$CONFIG")
DIAGNOSTICS_ROOT=$(awk '$1 == "diagnostics_root:" {print $2; exit}' "$CONFIG")
BOXER_DIAGNOSTICS_ROOT=$(awk '$1 == "diagnostics_dir:" {print $2; exit}' "$CONFIG")
[[ "$PRED_ROOT" == "$ROOT/results/$EXPERIMENT" ]] || {
  echo "Config output directory does not match the new experiment: $PRED_ROOT" >&2
  exit 1
}
[[ "$DIAGNOSTICS_ROOT" == \
  "$ROOT/diagnostics/boxer_tm_fpf_c1_stream3dv2_lightweight/route" ]] || {
  echo "Unexpected route diagnostics root: $DIAGNOSTICS_ROOT" >&2
  exit 1
}
[[ "$BOXER_DIAGNOSTICS_ROOT" == \
  "$ROOT/diagnostics/boxer_tm_fpf_c1_stream3dv2_lightweight/boxer" ]] || {
  echo "Unexpected Boxer diagnostics root: $BOXER_DIAGNOSTICS_ROOT" >&2
  exit 1
}

"$PYTHON" - "${BOXFUSION_PY_SOURCES[@]}" "$ROOT/demo.py" <<'PY'
from pathlib import Path
import sys

for value in sys.argv[1:]:
    path = Path(value)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print(f"Python syntax OK: {len(sys.argv) - 1} sources")
PY
bash -n "$RUNNER"

compute_source_fingerprint() {
  {
    sha256sum "${FINGERPRINT_SOURCES[@]}"
    printf '%s\n' \
      "python=$(readlink -f "$PYTHON")" \
      "python_version=$($PYTHON --version 2>&1)" \
      "scene_list_sha=$EXPECTED_OFFICIAL_LIST_SHA" \
      "boxer_commit=$EXPECTED_BOXER_COMMIT"
  } | sha256sum | awk '{print $1}'
}

FINGERPRINT="$(compute_source_fingerprint)"
if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  echo "Official100 lightweight preflight OK"
  echo "Scenes=100 GPU0=50 GPU1=50 fingerprint=$FINGERPRINT"
  exit 0
fi

LOG_ROOT="$ROOT/logs/$EXPERIMENT"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"
SUMMARY_PATH="$LOG_ROOT/OFFICIAL100_LIVE_SUMMARY.json"

mkdir -p \
  "$PRED_ROOT" "$DIAGNOSTICS_ROOT" "$BOXER_DIAGNOSTICS_ROOT" \
  "$SCENE_LOG_ROOT" "$MPL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
  echo "Another official100 lightweight driver holds $LOG_ROOT/run.lock" >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

CUDA_VISIBLE_DEVICES=0 \
PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
  'import torch, ultralytics, open_clip, yaml; assert torch.cuda.is_available(); print("Lightweight environment OK", torch.__version__, torch.version.cuda, ultralytics.__version__)'

echo "[$(date '+%F %T')] Lightweight Boxer + TM-FPF-C1 official100 started"
echo "[$(date '+%F %T')] GPUs=$GPU_SPEC scenes=100 shard_counts=50,50 fingerprint=$FINGERPRINT"
echo "[$(date '+%F %T')] Config=$CONFIG"
echo "[$(date '+%F %T')] Predictions=$PRED_ROOT"
echo "[$(date '+%F %T')] Route diagnostics=$DIAGNOSTICS_ROOT"

validate_completed_scene() {
  local scene="$1"
  local pred_path="$PRED_ROOT/${scene}_boxes.pkl"
  local diagnostic_path="$DIAGNOSTICS_ROOT/${scene}.json"
  local scene_log="$SCENE_LOG_ROOT/${scene}.log"
  local marker="$PRED_ROOT/${scene}.run_fingerprint"

  [[ -s "$pred_path" && -s "$diagnostic_path" && -s "$scene_log" && -s "$marker" ]] || return 1
  [[ "$(tr -d '\n' < "$marker")" == "$FINGERPRINT" ]] || return 1
  [[ "$(grep -Fc 'Strict live summary |' "$scene_log")" -eq 1 ]] || return 1
  [[ "$(grep -Ec 'Cost: [0-9.]+ s Average FPS: [0-9.]+' "$scene_log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|Exception in thread|STRICT LIVE FAILURE' "$scene_log" || return 1
  "$PYTHON" - "$diagnostic_path" "$scene" "$pred_path" <<'PY' >/dev/null 2>&1
import json
import pickle
import sys

import numpy as np

with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.load(handle)
assert row["schema"] == "boxfusion.stream3dv2_live.v1"
assert row["complete"] is True and row["scene_id"] == sys.argv[2]
for field in ("training_free", "past_only", "query_before_commit", "native_scores_preserved"):
    assert row[field] is True
for field in ("gt_access", "annotation_access", "evaluator_access", "proposal_cache_access", "teacher_cache_access", "terminal_cache_access"):
    assert row[field] is False
light = row["lightweight"]
assert light["enabled"] is True and light["depth_trigger_enabled"] is True
assert int(light["fastsam_box_shortlist"]) == 12
assert int(light["fastsam_top_k"]) == 8
assert light["conditional_f2"] is True
assert int(light["f4_top_m_tracks"]) == 8
assert light["terminal_clip_enabled"] is True
assert row["sam3"]["enabled"] is False
counts = row["counts"]
assert int(counts["output"]) == int(counts["native"]) + int(counts["births"])
with open(sys.argv[3], "rb") as handle:
    payload = pickle.load(handle)
assert isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], list)
for prediction in payload[0]:
    assert len(prediction) >= 3
    assert np.isfinite(np.asarray(prediction[0], dtype=np.float64)).all()
    assert np.isfinite(float(prediction[2])) and float(prediction[2]) > 0.0
PY
}

run_worker() {
  local gpu="$1"
  local shard="$2"
  local completed=0
  local index scene pred_path diagnostic_path scene_log marker

  for index in "${!SCENES[@]}"; do
    (( index % 2 == shard )) || continue
    scene="${SCENES[$index]}"
    pred_path="$PRED_ROOT/${scene}_boxes.pkl"
    diagnostic_path="$DIAGNOSTICS_ROOT/${scene}.json"
    scene_log="$SCENE_LOG_ROOT/${scene}.log"
    marker="$PRED_ROOT/${scene}.run_fingerprint"

    if validate_completed_scene "$scene"; then
      completed=$((completed + 1))
      echo "[$(date '+%F %T')] [GPU $gpu] Reusing same-fingerprint complete $scene"
      continue
    fi
    if [[ -e "$pred_path" || -e "$diagnostic_path" || -e "$scene_log" || -e "$marker" ]]; then
      echo "Partial, stale, or invalid artifact exists for $scene; refusing to overwrite" >&2
      return 1
    fi
    [[ "$(compute_source_fingerprint)" == "$FINGERPRINT" ]] || {
      echo "Source fingerprint changed before $scene; refusing a mixed run" >&2
      return 1
    }

    echo "[$(date '+%F %T')] [GPU $gpu] Running fresh $scene (official index $((index + 1))/100)"
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

    [[ -s "$pred_path" && -s "$diagnostic_path" ]] || {
      echo "Fresh run did not produce complete artifacts for $scene" >&2
      return 1
    }
    printf '%s\n' "$FINGERPRINT" >"$marker"
    validate_completed_scene "$scene" || {
      echo "Official100 scene validation failed for $scene; see $scene_log" >&2
      return 1
    }
    completed=$((completed + 1))
    echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $(grep -F 'Strict live summary |' "$scene_log")"
  done
  [[ "$completed" -eq 50 ]] || {
    echo "GPU $gpu shard completed $completed scenes instead of 50" >&2
    return 1
  }
  echo "[$(date '+%F %T')] [GPU $gpu] Completed exactly 50 scenes"
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

mkdir -p "$MPL_ROOT/worker0" "$MPL_ROOT/worker1"
run_worker 0 0 &
child_pids+=("$!")
run_worker 1 1 &
child_pids+=("$!")

worker_status=0
for pid in "${child_pids[@]}"; do
  if ! wait "$pid"; then
    worker_status=1
  fi
done
trap - INT TERM
[[ "$worker_status" -eq 0 ]] || {
  echo "At least one official100 worker failed; evaluation was not started" >&2
  exit 1
}

for scene in "${SCENES[@]}"; do
  validate_completed_scene "$scene" || {
    echo "Final artifact validation failed for $scene" >&2
    exit 1
  }
done
[[ "$(compute_source_fingerprint)" == "$FINGERPRINT" ]] || {
  echo "Source fingerprint changed before evaluation" >&2
  exit 1
}

prediction_count=$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
diagnostic_count=$(find "$DIAGNOSTICS_ROOT" -maxdepth 1 -type f -name 'scene*.json' | wc -l)
[[ "$prediction_count" -eq 100 ]] || {
  echo "Expected exactly 100 predictions, found $prediction_count" >&2
  exit 1
}
[[ "$diagnostic_count" -eq 100 ]] || {
  echo "Expected exactly 100 route diagnostics, found $diagnostic_count" >&2
  exit 1
}

echo "[$(date '+%F %T')] Starting official100 real-score AP evaluation"
bash "$EVAL_RUNNER" "$EXPERIMENT" "$PRED_ROOT"
echo "[$(date '+%F %T')] Aggregating official100 FPS and route diagnostics"
"$PYTHON" "$SUMMARY_TOOL" \
  --scene-list "$SCENE_LIST" \
  --diagnostics-root "$DIAGNOSTICS_ROOT" \
  --scene-log-root "$SCENE_LOG_ROOT" \
  --output "$SUMMARY_PATH" \
  --require-complete
echo "[$(date '+%F %T')] Lightweight Boxer + TM-FPF-C1 official100 complete"
