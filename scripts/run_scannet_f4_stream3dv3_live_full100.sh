#!/usr/bin/env bash
set -euo pipefail

# Strict fresh-inference Stream3Dv3 official100 run.
# Usage: bash scripts/run_scannet_f4_stream3dv3_live_full100.sh 0[,1,...]

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 GPU[,GPU,...]" >&2
  exit 2
fi

GPU_SPEC="$1"
ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
CONFIG="$ROOT/config/scannet_cbest_f4_stream3dv3_live_score05.yaml"
OFFICIAL_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
SCENE_LIST="${BOXFUSION_V3_SCENE_LIST:-$OFFICIAL_SCENE_LIST}"
SKIP_EVAL="${BOXFUSION_V3_SKIP_EVAL:-0}"
PREFLIGHT_ONLY="${BOXFUSION_V3_PREFLIGHT_ONLY:-0}"
EXPERIMENT=scannet_cbest_f4_stream3dv3_live_score05
RUNNER="$ROOT/scripts/run_scannet_f4_stream3dv3_live_full100.sh"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
SUMMARY_TOOL="$ROOT/tools/summarize_stream3dv3_live_official100.py"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$ROOT/data/class_features.pt"
FASTSAM_CHECKPOINT=/data/ZhaoX/RoboFusion/RoboFusion-master/focalconvsamfusion/OpenPCDet/checkpoints/FastSAM.pt
BOXER_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer
BOXER_CHECKPOINT="$BOXER_ROOT/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CHECKPOINT="$BOXER_ROOT/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
EXPECTED_OFFICIAL_LIST_SHA=4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5
EXPECTED_BOXER_COMMIT=1f86542dc342a4b1d474c87c97c5d1d6566d9148
EXPECTED_BOXER_SHA=d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f
EXPECTED_DINO_SHA=4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea
EXPECTED_MODEL_SHA=856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217
EXPECTED_CLIP_SHA=9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4
EXPECTED_FASTSAM_SHA=c0be4e7ddbe4c15333d15a859c676d053c486d0a746a3be6a7a9790d52a9b6d7
EXPECTED_CLASS_TXT_SHA=0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9
EXPECTED_CLASS_FEATURES_SHA=49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197
EXPECTED_PST_SHA=867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

V3_SOURCES=(
  "$ROOT/demo.py"
  "$ROOT/boxfusion/stream3dv3_trigger.py"
  "$ROOT/boxfusion/stream3dv3_track_fusion.py"
  "$ROOT/boxfusion/stream3dv3_live.py"
  "$ROOT/boxfusion/stream3dv2_lite.py"
  "$ROOT/boxfusion/fastsam_automatic_provider.py"
  "$ROOT/boxfusion/fastsam_residual_shadow.py"
  "$ROOT/boxfusion/fastsam_dfu_lgf_shadow.py"
  "$ROOT/boxfusion/fastsam_openbox_f3_shadow.py"
  "$ROOT/boxfusion/fastsam_boxer_f4_shadow.py"
  "$ROOT/boxfusion/boxer_lifter.py"
  "$ROOT/boxfusion/box_fusion.py"
  "$ROOT/boxfusion/box_manager.py"
  "$ROOT/boxfusion/instances.py"
  "$ROOT/boxfusion/boxes.py"
  "$ROOT/boxfusion/reliable_views.py"
)
mapfile -t BOXFUSION_PY_SOURCES < <(
  rg --files "$ROOT/boxfusion" | rg '\.py$' | sort
)
FINGERPRINT_SOURCES=(
  "$ROOT/demo.py"
  "$ROOT/tools/utils.py"
  "${BOXFUSION_PY_SOURCES[@]}"
)

for required in \
  "$PYTHON" "$CONFIG" "$SCENE_LIST" "$RUNNER" "$EVAL_RUNNER" "$SUMMARY_TOOL" \
  "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" "$CLASS_FEATURES" "$FASTSAM_CHECKPOINT" \
  "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT" "${FINGERPRINT_SOURCES[@]}"; do
  [[ -e "$required" ]] || {
    echo "Missing required V3 input: $required" >&2
    exit 1
  }
done

grep -Fq 'from boxfusion.stream3dv3_live import build_stream3dv3_live_route' "$ROOT/demo.py" || {
  echo "demo.py has not imported the Stream3Dv3 route" >&2
  exit 1
}
grep -Fq 'build_stream3dv3_live_route(' "$ROOT/demo.py" || {
  echo "demo.py has not built the Stream3Dv3 route" >&2
  exit 1
}
for required_call in \
  'stream3dv3_live.start_pipeline_clock(' \
  'strict_live_route.poll(' \
  'strict_live_route.process_keyframe(' \
  'strict_live_route.finalize(' \
  '"Stream3Dv3 live summary |"'; do
  grep -Fq "$required_call" "$ROOT/demo.py" || {
    echo "demo.py is missing the Stream3Dv3 live call: $required_call" >&2
    exit 1
  }
done

for switch in "$SKIP_EVAL" "$PREFLIGHT_ONLY"; do
  case "$switch" in
    0|1) ;;
    *)
      echo "BOXFUSION_V3_SKIP_EVAL and BOXFUSION_V3_PREFLIGHT_ONLY must be 0 or 1" >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "No GPU was specified" >&2; exit 2; }
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $gpu" >&2; exit 2; }
  [[ -z "${SEEN_GPUS[$gpu]:-}" ]] || { echo "Duplicate GPU index: $gpu" >&2; exit 2; }
  SEEN_GPUS[$gpu]=1
done

[[ "$(git -C "$BOXER_ROOT" rev-parse HEAD)" == "$EXPECTED_BOXER_COMMIT" ]] || {
  echo "Boxer commit mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$BOXER_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_BOXER_SHA" ]] || {
  echo "Boxer checkpoint mismatch" >&2
  exit 1
}
[[ "$(sha256sum "$DINO_CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_DINO_SHA" ]] || {
  echo "DINO checkpoint mismatch" >&2
  exit 1
}
for asset_and_hash in \
  "$MODEL:$EXPECTED_MODEL_SHA" \
  "$CLIP_MODEL:$EXPECTED_CLIP_SHA" \
  "$FASTSAM_CHECKPOINT:$EXPECTED_FASTSAM_SHA" \
  "$CLASS_TXT:$EXPECTED_CLASS_TXT_SHA" \
  "$CLASS_FEATURES:$EXPECTED_CLASS_FEATURES_SHA" \
  "$ROOT/data/pst_1024_0.tiff:$EXPECTED_PST_SHA"; do
  asset="${asset_and_hash%:*}"
  expected="${asset_and_hash##*:}"
  [[ "$(sha256sum "$asset" | awk '{print $1}')" == "$expected" ]] || {
    echo "Asset hash mismatch: $asset" >&2
    exit 1
  }
done

"$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

assert cfg["dataset"] == "scannet"
assert cfg["data"]["gap"] == 25
assert cfg["detection"]["score_thresh"] == 0.5
assert cfg["lifting"]["backend"] == "boxer"
assert cfg["lifting"]["boxer"]["mode"] == "active"
assert cfg["lifting"]["boxer"]["apply_stage"] == "post_filter"
proposal_cache = cfg["lifting"].get("proposal_cache", {})
assert str(proposal_cache.get("mode", "disabled")).lower() in {"", "disabled", "none", "off"}
assert cfg["association"]["appearance_gate"]["enabled"] is False
assert cfg["box_fusion"]["vapf_lite"]["enabled"] is False
assert cfg.get("online_stream3dv2", {}).get("enabled", False) is False
v3 = cfg["online_stream3dv3"]
assert v3["enabled"] is True and v3["strict_fresh"] is True
assert "sam3" not in v3
assert float(v3["target_end_to_end_fps"]) >= 20.0
assert float(v3["addon_deadline_ms"]) == 285.0
assert int(v3["f0"]["prelift_top_k"]) == 6
assert int(v3["f4"]["max_views_per_track"]) == 2
assert int(v3["output"]["max_births_per_scene"]) == 2
assert int(v3["output"]["max_overlays_per_scene"]) == 0
assert v3["output"]["allow_abstain"] is True
assert cfg["box_fusion"]["reliable_views"]["enabled"] is True
assert int(cfg["box_fusion"]["reliable_views"]["top_k"]) == 3
assert cfg["eval"] is True
assert cfg["vis"]["rerun"] is False
print("Stream3Dv3 strict-fresh protocol semantics OK")
PY

PRED_ROOT=$(awk '$1 == "output_dir:" {print $2; exit}' "$CONFIG")
DIAGNOSTICS_ROOT=$(awk '$1 == "diagnostics_root:" {print $2; exit}' "$CONFIG")
[[ "$PRED_ROOT" == "$ROOT/results/$EXPERIMENT" ]] || {
  echo "Config output directory is not the sealed V3 experiment root: $PRED_ROOT" >&2
  exit 1
}
[[ "$DIAGNOSTICS_ROOT" == "$ROOT/diagnostics/cbest_f4_stream3dv3_live/route" ]] || {
  echo "Config diagnostics directory is not the sealed V3 route root: $DIAGNOSTICS_ROOT" >&2
  exit 1
}

if [[ "$SCENE_LIST" == "$OFFICIAL_SCENE_LIST" ]]; then
  [[ "$(sha256sum "$OFFICIAL_SCENE_LIST" | awk '{print $1}')" == "$EXPECTED_OFFICIAL_LIST_SHA" ]] || {
    echo "Official100 scene-list hash mismatch" >&2
    exit 1
  }
fi

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -gt 0 ]] || { echo "Scene list is empty: $SCENE_LIST" >&2; exit 1; }
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
  echo "A custom BOXFUSION_V3_SCENE_LIST requires BOXFUSION_V3_SKIP_EVAL=1" >&2
  exit 2
fi

"$PYTHON" - "${V3_SOURCES[@]}" <<'PY'
from pathlib import Path
import sys

for value in sys.argv[1:]:
    path = Path(value)
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
print(f"Python syntax OK: {len(sys.argv) - 1} sources")
PY
bash -n "$0"

compute_fingerprint() {
  {
    sha256sum \
      "$CONFIG" "$SCENE_LIST" "$RUNNER" "$EVAL_RUNNER" "$SUMMARY_TOOL" \
      "${FINGERPRINT_SOURCES[@]}" \
      "$ROOT/data/pst_1024_0.tiff" "$MODEL" "$CLIP_MODEL" \
      "$CLASS_TXT" "$CLASS_FEATURES" "$FASTSAM_CHECKPOINT" \
      "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT"
    printf '%s\n' \
      "python=$(readlink -f "$PYTHON")" \
      "python_version=$($PYTHON --version 2>&1)" \
      "boxer_commit=$EXPECTED_BOXER_COMMIT"
  } | sha256sum | awk '{print $1}'
}

FINGERPRINT="$(compute_fingerprint)"

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  echo "Stream3Dv3 preflight OK"
  echo "Scenes=${#SCENES[@]} GPUs=$GPU_SPEC fingerprint=$FINGERPRINT"
  exit 0
fi

LOG_ROOT="$ROOT/logs/$EXPERIMENT"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MPL_ROOT="$LOG_ROOT/mplconfig"
SUMMARY_PATH="$LOG_ROOT/OFFICIAL100_LIVE_SUMMARY.json"

mkdir -p "$PRED_ROOT" "$DIAGNOSTICS_ROOT" "$SCENE_LOG_ROOT" "$MPL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
  echo "Another V3 driver holds $LOG_ROOT/run.lock" >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
  'import torch, ultralytics, open_clip, yaml; assert torch.cuda.is_available(); assert ultralytics.__version__ == "8.4.105"; print("V3 environment OK", torch.__version__, torch.version.cuda, ultralytics.__version__)'

echo "[$(date '+%F %T')] F4 Stream3Dv3 strict-fresh inference started"
echo "[$(date '+%F %T')] GPUs=$GPU_SPEC workers=${#GPUS[@]} scenes=${#SCENES[@]} fingerprint=$FINGERPRINT"

validate_completed_scene() {
  local scene="$1"
  local pred_path="$PRED_ROOT/${scene}_boxes.pkl"
  local diagnostic_path="$DIAGNOSTICS_ROOT/${scene}.json"
  local scene_log="$SCENE_LOG_ROOT/${scene}.log"
  local marker="$PRED_ROOT/${scene}.run_fingerprint"

  [[ -s "$pred_path" && -s "$diagnostic_path" && -s "$scene_log" && -s "$marker" ]] || return 1
  [[ "$(tr -d '\n' < "$marker")" == "$FINGERPRINT" ]] || return 1
  [[ "$(grep -Fc 'Stream3Dv3 live summary |' "$scene_log")" -eq 1 ]] || return 1
  [[ "$(grep -Ec 'Cost: [0-9.]+ s Average FPS: [0-9.]+' "$scene_log")" -eq 1 ]] || return 1
  ! grep -Eq 'Traceback|Exception in thread|STREAM3DV3 FAILURE' "$scene_log" || return 1
  "$PYTHON" - "$diagnostic_path" "$scene" "$pred_path" "$FINGERPRINT" <<'PY' >/dev/null 2>&1
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.load(handle)
assert row["schema"] == "boxfusion.stream3dv3_live.v1"
assert row["complete"] is True and row["scene_id"] == sys.argv[2]
assert row["run_fingerprint"] == sys.argv[4]
assert re.fullmatch(r"[0-9a-f]{64}", row["run_fingerprint"])
assert row["fresh_inference"] is True
assert row["training_free"] is True
assert row["pose_source"] == "scannet_provided_pose"
assert row["past_only"] is True and row["query_before_commit"] is True
assert row["selection_and_acceptance_held_out"] is True
assert row["future_access_count"] == 0
for field in ("ground_truth_access", "annotation_access", "evaluator_access", "proposal_cache_access", "teacher_cache_access", "terminal_cache_access"):
    assert row[field] is False
assert row["native_scores_preserved"] is True
assert float(row["target_end_to_end_fps"]) >= 20.0
counts = row["counts"]
assert 0 <= int(counts["births"]) <= 2
assert int(counts["overlays"]) == 0
assert int(counts["output"]) == int(counts["native"]) + int(counts["births"])
bounded = row["bounded"]
assert int(bounded["max_births_per_scene"]) == 2
assert int(bounded["max_f4_views_per_track"]) == 2
assert int(bounded["max_f4_attempts_observed"]) <= 2
assert int(bounded["max_accepted_tracks"]) == 128
assert int(bounded["prelift_top_k"]) == 6
assert all(int(value) <= 2 for value in row["f4_per_track"].values())
assert row["sam3"] == {"enabled": False}
assert int(row["raw_frame_count"]) > 0
assert float(row["pipeline_seconds"]) > 0.0
schedule = row["schedule"]
frame_root = Path("/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames") / sys.argv[2] / "frames"
source_counts = {
    "color": len(list((frame_root / "color").glob("*.jpg"))),
    "depth": len(list((frame_root / "depth").glob("*.png"))),
    "pose": len(list((frame_root / "pose").glob("*.txt"))),
}
assert len(set(source_counts.values())) == 1 and source_counts["color"] > 0
dataset_frames = source_counts["color"]
expected_raw = max(1, dataset_frames - 25)
expected_keyframes = (expected_raw + 24) // 25
assert int(schedule["dataset_frame_count"]) == dataset_frames
assert int(schedule["keyframe_gap"]) == 25
assert int(schedule["expected_raw_frame_count"]) == expected_raw
assert int(schedule["expected_keyframe_count"]) == expected_keyframes
assert int(row["raw_frame_count"]) == expected_raw
assert int(counts["keyframes"]) == expected_keyframes
assert int(row["f3"]["keyframes"]) == expected_keyframes

with open(sys.argv[3], "rb") as handle:
    payload = pickle.load(handle)
assert isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], list)
prediction_rows = payload[0]
assert len(prediction_rows) == int(counts["output"])
corners = (
    np.asarray([item[1] for item in prediction_rows], dtype=np.float32)
    if prediction_rows
    else np.empty((0, 8, 3), dtype=np.float32)
)
scores = np.asarray([item[2] for item in prediction_rows], dtype=np.float32)
assert corners.shape == (len(scores), 8, 3)
assert np.isfinite(corners).all() and np.isfinite(scores).all()
assert np.all(scores > 0.05)
native_count = int(counts["native"])
native_scores = np.ascontiguousarray(scores[:native_count], dtype=np.float32)
prefix_sha = hashlib.sha256(native_scores.tobytes()).hexdigest()
score_audit = row["score_audit"]
assert score_audit["dtype"] == "float32"
assert prefix_sha == score_audit["native_prefix_sha256"]
assert prefix_sha == score_audit["output_prefix_sha256"]
native_corners = np.ascontiguousarray(corners[:native_count], dtype=np.float32)
corners_sha = hashlib.sha256(native_corners.tobytes()).hexdigest()
geometry_audit = row["geometry_audit"]
assert geometry_audit["dtype"] == "float32"
assert corners_sha == geometry_audit["native_prefix_sha256"]
assert corners_sha == geometry_audit["output_prefix_sha256"]
append_scores = np.ascontiguousarray(scores[native_count:], dtype=np.float32)
assert np.array_equal(
    append_scores,
    np.asarray(score_audit["append_scores"], dtype=np.float32),
)
assert len(np.unique(append_scores)) == len(append_scores)
if len(append_scores) and len(native_scores):
    assert np.all(append_scores < np.min(native_scores))
PY
}

run_worker() {
  local gpu="$1"
  local shard="$2"
  local shard_count="$3"
  local completed=0
  local index scene pred_path diagnostic_path scene_log marker

  for index in "${!SCENES[@]}"; do
    (( index % shard_count == shard )) || continue
    scene="${SCENES[$index]}"
    pred_path="$PRED_ROOT/${scene}_boxes.pkl"
    diagnostic_path="$DIAGNOSTICS_ROOT/${scene}.json"
    scene_log="$SCENE_LOG_ROOT/${scene}.log"
    marker="$PRED_ROOT/${scene}.run_fingerprint"

    [[ "$(compute_fingerprint)" == "$FINGERPRINT" ]] || {
      echo "V3 source/asset fingerprint changed before $scene; refusing a mixed run" >&2
      return 1
    }

    if validate_completed_scene "$scene"; then
      completed=$((completed + 1))
      echo "[$(date '+%F %T')] [GPU $gpu] Reusing complete $scene"
      continue
    fi
    if [[ -e "$pred_path" || -e "$diagnostic_path" || -e "$scene_log" || -e "$marker" ]]; then
      echo "Partial, stale, or invalid V3 artifact exists for $scene; refusing to overwrite" >&2
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
      HF_HUB_OFFLINE=1 \
      MPLCONFIGDIR="$MPL_ROOT/worker${shard}" \
      LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
      BOXFUSION_STRICT_LIVE=1 \
      BOXFUSION_RUN_FINGERPRINT="$FINGERPRINT" \
      OMP_NUM_THREADS=8 \
      "$PYTHON" demo.py scannet \
        --model-path "$MODEL" \
        --clip_path "$CLIP_MODEL" \
        --class_txt "$CLASS_TXT" \
        --config "$CONFIG" \
        --device cuda \
        --seq "$scene"
    ) >"$scene_log" 2>&1

    [[ "$(compute_fingerprint)" == "$FINGERPRINT" ]] || {
      echo "V3 source/asset fingerprint changed while running $scene" >&2
      return 1
    }

    [[ -s "$pred_path" && -s "$diagnostic_path" ]] || {
      echo "V3 scene failed to produce outputs: $scene; see $scene_log" >&2
      return 1
    }
    printf '%s\n' "$FINGERPRINT" >"${marker}.tmp.$$"
    mv "${marker}.tmp.$$" "$marker"
    if ! validate_completed_scene "$scene"; then
      echo "V3 artifact validation failed for $scene; see $scene_log" >&2
      return 1
    fi
    completed=$((completed + 1))
    echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $(grep -F 'Stream3Dv3 live summary |' "$scene_log" | tail -n 1)"
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
[[ "$worker_status" -eq 0 ]] || {
  echo "At least one V3 worker failed; evaluation was not started" >&2
  exit 1
}

for scene in "${SCENES[@]}"; do
  validate_completed_scene "$scene" || {
    echo "Final V3 artifact validation failed for $scene" >&2
    exit 1
  }
done

[[ "$(compute_fingerprint)" == "$FINGERPRINT" ]] || {
  echo "V3 source/asset fingerprint changed before evaluation" >&2
  exit 1
}

echo "[$(date '+%F %T')] All ${#SCENES[@]} requested V3 scenes passed artifact validation"
if [[ "$SKIP_EVAL" -eq 1 ]]; then
  echo "[$(date '+%F %T')] Evaluation and official100 summary skipped"
  exit 0
fi

prediction_count=$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
diagnostic_count=$(find "$DIAGNOSTICS_ROOT" -maxdepth 1 -type f -name 'scene*.json' | wc -l)
[[ "$prediction_count" -eq 100 ]] || { echo "Expected exactly 100 predictions, found $prediction_count" >&2; exit 1; }
[[ "$diagnostic_count" -eq 100 ]] || { echo "Expected exactly 100 diagnostics, found $diagnostic_count" >&2; exit 1; }

echo "[$(date '+%F %T')] Starting official100 real-score AP evaluation"
bash "$EVAL_RUNNER" "$EXPERIMENT" "$PRED_ROOT"
"$PYTHON" "$SUMMARY_TOOL" \
  --scene-list "$SCENE_LIST" \
  --diagnostics-root "$DIAGNOSTICS_ROOT" \
  --output "$SUMMARY_PATH" \
  --minimum-fps 20.0 \
  --require-complete \
  --require-realtime-pass
echo "[$(date '+%F %T')] F4 Stream3Dv3 strict-fresh official100 complete"
