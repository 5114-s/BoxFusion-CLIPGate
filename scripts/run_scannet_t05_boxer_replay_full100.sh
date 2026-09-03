#!/usr/bin/env bash
set -euo pipefail

# Paired Top-K + Boxer geometry experiment on the sealed score-0.5 CuTR cache.
#
# Usage:
#   bash scripts/run_scannet_t05_boxer_replay_full100.sh observer 0,1
#   bash scripts/run_scannet_t05_boxer_replay_full100.sh active   0,1
#
# Optional smoke/resume controls:
#   BOXFUSION_T05_BOXER_SCENE_LIST=/path/to/list.txt
#   BOXFUSION_T05_BOXER_SKIP_EVAL=1

PROFILE="${1:-}"
GPU_SPEC="${2:-0,1}"

ROOT=/data/ZhaoX/BoxFusion
EXEC_ROOT="$ROOT/tools/boxfusion_experiment_archive/snapshots/boxfusion_boxer_dev/source"
EXEC_DEMO="$EXEC_ROOT/demo.py"
ARCHIVE_MANIFEST="$(dirname "$EXEC_ROOT")/MANIFEST.json"
EXPECTED_ARCHIVE_MANIFEST_SHA=1c60104f7089e10c562b51350d9b704473de235f6d204e66c2dc3a86a05f4611
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="${BOXFUSION_T05_BOXER_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
SKIP_EVAL="${BOXFUSION_T05_BOXER_SKIP_EVAL:-0}"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$ROOT/data/class_features.pt"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_cgf_paper100_constant_score.sh"
ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

case "$PROFILE" in
  observer)
    CONFIG="$ROOT/config/scannet_t05_boxer_replay_observer_score05.yaml"
    EXPERIMENT_NAME=scannet_t05_boxer_replay_observer_score05
    ;;
  active)
    CONFIG="$ROOT/config/scannet_t05_boxer_replay_active_score05.yaml"
    EXPERIMENT_NAME=scannet_t05_boxer_replay_active_score05
    ;;
  *)
    echo "Profile must be observer or active" >&2
    exit 2
    ;;
esac

for required in \
  "$PYTHON" \
  "$SCENE_LIST" \
  "$CONFIG" \
  "$EXEC_DEMO" \
  "$ARCHIVE_MANIFEST" \
  "$CLASS_TXT" \
  "$CLASS_FEATURES" \
  "$MODEL" \
  "$CLIP_MODEL" \
  "$EVAL_RUNNER"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required path: $required" >&2
    exit 1
  fi
done

verify_archive_manifest() {
  local actual_manifest_sha
  actual_manifest_sha="$(sha256sum "$ARCHIVE_MANIFEST" | awk '{print $1}')"
  if [[ "$actual_manifest_sha" != "$EXPECTED_ARCHIVE_MANIFEST_SHA" ]]; then
    echo "Archived source manifest SHA mismatch: $actual_manifest_sha" >&2
    return 1
  fi
  "$PYTHON" - "$EXEC_ROOT" "$ARCHIVE_MANIFEST" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source_root = Path(sys.argv[1])
with Path(sys.argv[2]).open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
expected = manifest["files"]
actual = {
    path.relative_to(source_root).as_posix()
    for path in source_root.rglob("*")
    if path.is_file()
}
if actual != set(expected):
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    raise SystemExit(f"Archive file-set mismatch; missing={missing}, extra={extra}")
for relative, record in expected.items():
    digest = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise SystemExit(f"Archive content mismatch: {relative}: {digest}")
print(f"Archive integrity OK: {len(expected)}/{len(expected)} files")
PY
}

verify_archive_manifest

read_config_value() {
  local expression="$1"
  "$PYTHON" - "$CONFIG" "$expression" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    value = yaml.safe_load(handle)
for key in sys.argv[2].split("."):
    value = value[key]
print(value)
PY
}

PRED_ROOT="$(read_config_value data.output_dir)"
DIAGNOSTICS_ROOT="$(read_config_value lifting.boxer.diagnostics_dir)"
CACHE_ROOT="$(read_config_value lifting.proposal_cache.root)"
CACHE_NAMESPACE="$(read_config_value lifting.proposal_cache.namespace)"
CACHE_BASELINE_ROOT="$(read_config_value lifting.proposal_cache.baseline_prediction_root)"
BOXER_ROOT="$(read_config_value lifting.boxer.official_root)"
BOXER_CHECKPOINT="$(read_config_value lifting.boxer.checkpoint)"
BOXER_EXPECTED_COMMIT="$(read_config_value lifting.boxer.expected_commit)"
BOXER_CHECKPOINT_SHA="$(read_config_value lifting.boxer.checkpoint_sha256)"
DINO_EXPECTED_SHA="$(read_config_value lifting.boxer.dinov3_sha256)"
DINO_CHECKPOINT="$BOXER_ROOT/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

"$PYTHON" - "$CONFIG" "$PROFILE" <<'PY'
import sys
import yaml

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
expected = {
    "dataset": "scannet",
    "gap": 25,
    "score_thresh": 0.5,
    "backend": "boxer",
    "cache_mode": "replay",
    "apply_stage": "post_filter",
    "boxer_mode": sys.argv[2],
    "appearance_gate": False,
    "topk_enabled": True,
    "top_k": 3,
    "min_views": 3,
}
actual = {
    "dataset": cfg["dataset"],
    "gap": cfg["data"]["gap"],
    "score_thresh": cfg["detection"]["score_thresh"],
    "backend": cfg["lifting"]["backend"],
    "cache_mode": cfg["lifting"]["proposal_cache"]["mode"],
    "apply_stage": cfg["lifting"]["boxer"]["apply_stage"],
    "boxer_mode": cfg["lifting"]["boxer"]["mode"],
    "appearance_gate": cfg["association"]["appearance_gate"]["enabled"],
    "topk_enabled": cfg["box_fusion"]["reliable_views"]["enabled"],
    "top_k": cfg["box_fusion"]["reliable_views"]["top_k"],
    "min_views": cfg["box_fusion"]["reliable_views"]["min_views"],
}
if actual != expected:
    raise SystemExit(f"Experiment protocol mismatch: expected={expected}, actual={actual}")
print(f"Protocol semantics OK: {actual}")
PY

for required in \
  "$CACHE_ROOT/$CACHE_NAMESPACE" \
  "$CACHE_BASELINE_ROOT" \
  "$BOXER_ROOT" \
  "$BOXER_CHECKPOINT" \
  "$DINO_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing sealed experiment asset: $required" >&2
    exit 1
  fi
done

actual_boxer_commit="$(git -C "$BOXER_ROOT" rev-parse HEAD)"
actual_boxer_sha="$(sha256sum "$BOXER_CHECKPOINT" | awk '{print $1}')"
actual_dino_sha="$(sha256sum "$DINO_CHECKPOINT" | awk '{print $1}')"
if [[ "$actual_boxer_commit" != "$BOXER_EXPECTED_COMMIT" ]]; then
  echo "Boxer commit mismatch: $actual_boxer_commit" >&2
  exit 1
fi
if [[ "$actual_boxer_sha" != "$BOXER_CHECKPOINT_SHA" ]]; then
  echo "Boxer checkpoint SHA mismatch: $actual_boxer_sha" >&2
  exit 1
fi
if [[ "$actual_dino_sha" != "$DINO_EXPECTED_SHA" ]]; then
  echo "DINO checkpoint SHA mismatch: $actual_dino_sha" >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "At least one GPU is required" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index: $gpu" >&2
    exit 2
  fi
done
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
  echo "Duplicate GPU index in GPU specification: $GPU_SPEC" >&2
  exit 2
fi

duplicate_scene="$({ awk 'NF && $1 !~ /^#/ {count[$1] += 1} END {for (scene in count) if (count[scene] > 1) {print scene; exit}}' "$SCENE_LIST"; })"
if [[ -n "$duplicate_scene" ]]; then
  echo "Duplicate scene in scene list: $duplicate_scene" >&2
  exit 1
fi

LIST_SHA256="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
LIST_TAG="$(basename "$SCENE_LIST" .txt)-${LIST_SHA256:0:12}"
LOG_ROOT="$ROOT/logs/t05_boxer_replay/$PROFILE/$LIST_TAG"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
mkdir -p "$PRED_ROOT" "$DIAGNOSTICS_ROOT" "$SCENE_LOG_ROOT"

LOCK_ROOT="$ROOT/logs/t05_boxer_replay/locks"
mkdir -p "$LOCK_ROOT"
exec 9>"$LOCK_ROOT/${PROFILE}.lock"
if ! flock -n 9; then
  echo "Another $PROFILE runner is already active" >&2
  exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

GLOBAL_FINGERPRINT="$({
  sha256sum \
    "$CONFIG" \
    "$EXEC_DEMO" \
    "$EXEC_ROOT/boxfusion/boxer_lifter.py" \
    "$EXEC_ROOT/boxfusion/proposal_cache.py" \
    "$EXEC_ROOT/boxfusion/reliable_views.py" \
    "$EXEC_ROOT/boxfusion/box_fusion.py" \
    "$EXEC_ROOT/boxfusion/box_manager.py" \
    "$EXEC_ROOT/boxfusion/cubify_transformer.py" \
    "$EXEC_ROOT/boxfusion/instances.py" \
    "$EXEC_ROOT/boxfusion/boxes.py" \
    "$EXEC_ROOT/boxfusion/capture_stream.py" \
    "$EXEC_ROOT/boxfusion/preprocessor.py" \
    "$EXEC_ROOT/tools/utils.py" \
    "$ROOT/data/pst_1024_0.tiff" \
    "$CLASS_TXT" \
    "$CLASS_FEATURES" \
    "$MODEL" \
    "$CLIP_MODEL" \
    "$BOXER_CHECKPOINT" \
    "$DINO_CHECKPOINT"
  printf '%s\n' \
    "boxer_commit=$actual_boxer_commit" \
    "python=$(readlink -f "$PYTHON")" \
    "python_version=$($PYTHON --version 2>&1)"
} | sha256sum | awk '{print $1}')"

total="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
echo "[$(date '+%F %T')] Starting paired T05 + Boxer replay"
echo "[$(date '+%F %T')] profile=$PROFILE scenes=$total GPUs=$GPU_SPEC"
echo "[$(date '+%F %T')] executable=$EXEC_DEMO"
echo "[$(date '+%F %T')] config=$CONFIG"
echo "[$(date '+%F %T')] predictions=$PRED_ROOT"
echo "[$(date '+%F %T')] cache=$CACHE_ROOT/$CACHE_NAMESPACE"
echo "[$(date '+%F %T')] global_fingerprint=$GLOBAL_FINGERPRINT"
echo "[$(date '+%F %T')] scene_list_sha256=$LIST_SHA256"
echo "[$(date '+%F %T')] runner_sha256=$(sha256sum "$ROOT/scripts/run_scannet_t05_boxer_replay_full100.sh" | awk '{print $1}')"
echo "[$(date '+%F %T')] archive_manifest_sha256=$EXPECTED_ARCHIVE_MANIFEST_SHA"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c "import torch, torchvision, open_clip; assert torch.cuda.is_available(); print('Environment OK:', torch.__version__, torch.version.cuda)"

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
    local diagnostic_path="$DIAGNOSTICS_ROOT/${scene}_boxer_lifting.jsonl"
    local scene_log="$SCENE_LOG_ROOT/${scene}.log"
    local cache_manifest="$CACHE_ROOT/$CACHE_NAMESPACE/$scene/manifest.json"
    local baseline_marker="$CACHE_BASELINE_ROOT/${scene}.run_fingerprint"
    local scene_frames_root
    local input_fingerprint
    local producer_fingerprint
    local manifest_producer
    local scene_fingerprint

    scene_frames_root="$($PYTHON - "$CONFIG" "$scene" <<'PY'
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
    if [[ ! -s "$cache_manifest" || ! -s "$baseline_marker" ]]; then
      echo "Missing sealed replay cache or producer marker: $scene" >&2
      return 1
    fi

    producer_fingerprint="$(tr -d '\n' < "$baseline_marker")"
    manifest_producer="$($PYTHON - "$cache_manifest" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["producer_fingerprint"])
PY
)"
    if [[ "$manifest_producer" != "$producer_fingerprint" ]]; then
      echo "Cache producer mismatch for $scene" >&2
      return 1
    fi

    input_fingerprint="$(find "$scene_frames_root" -type f -printf '%P\t%s\t%T@\n' | LC_ALL=C sort | sha256sum | awk '{print $1}')"
    scene_fingerprint="$(printf '%s\n%s\n%s\n' "$GLOBAL_FINGERPRINT" "$input_fingerprint" "$(sha256sum "$cache_manifest" | awk '{print $1}')" | sha256sum | awk '{print $1}')"

    if [[ -s "$pred_path" ]]; then
      if [[ ! -s "$marker_path" || "$(tr -d '\n' < "$marker_path")" != "$scene_fingerprint" ]]; then
        echo "Refusing stale or untracked prediction: $pred_path" >&2
        return 1
      fi
      if [[ ! -s "$diagnostic_path" ]]; then
        echo "Prediction exists but Boxer diagnostic is missing: $diagnostic_path" >&2
        return 1
      fi
      completed=$((completed + 1))
      echo "[$(date '+%F %T')] [GPU $gpu] $scene already complete"
      index=$((index + 1))
      continue
    fi
    if [[ -e "$marker_path" || -e "$diagnostic_path" ]]; then
      echo "Refusing orphan marker/diagnostic for $scene" >&2
      return 1
    fi

    echo "[$(date '+%F %T')] [GPU $gpu] Running $scene (list index $((index + 1))/$total)"
    (
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=0 \
      PYTHONNOUSERSITE=1 \
      PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 \
      BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$producer_fingerprint" \
      OMP_NUM_THREADS=8 \
      MPLCONFIGDIR="$mpl_dir" \
      LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
      "$PYTHON" "$EXEC_DEMO" scannet \
        --model-path "$MODEL" \
        --clip_path "$CLIP_MODEL" \
        --class_txt "$CLASS_TXT" \
        --config "$CONFIG" \
        --device cuda \
        --seq "$scene"
    ) >"$scene_log" 2>&1

    if [[ ! -s "$pred_path" ]]; then
      echo "GPU $gpu did not produce $pred_path" >&2
      return 1
    fi
    if [[ ! -s "$diagnostic_path" ]]; then
      echo "GPU $gpu did not produce $diagnostic_path" >&2
      return 1
    fi
    printf '%s\n' "$scene_fingerprint" >"${marker_path}.tmp.$$"
    mv "${marker_path}.tmp.$$" "$marker_path"
    completed=$((completed + 1))

    local summary
    summary="$(rg 'Reliable-view fusion summary|Boxer lifting summary|Saving score-preserving predictions' "$scene_log" | tail -n 3 | tr '\n' ' ' || true)"
    echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $summary"
    index=$((index + 1))
  done < "$SCENE_LIST"
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

worker_count="${#GPUS[@]}"
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
  if [[ ! -s "$PRED_ROOT/${scene}_boxes.pkl" || ! -s "$DIAGNOSTICS_ROOT/${scene}_boxer_lifting.jsonl" ]]; then
    echo "Missing requested prediction/diagnostic: $scene" >&2
    exit 1
  fi
done < "$SCENE_LIST"

echo "[$(date '+%F %T')] Inference complete: $PROFILE ($total scenes)"
verify_archive_manifest
if [[ "$SKIP_EVAL" == "1" ]]; then
  echo "[$(date '+%F %T')] Constant-score evaluation skipped by request"
  exit 0
fi
if [[ "$LIST_SHA256" != "$(sha256sum "$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt" | awk '{print $1}')" ]]; then
  echo "Refusing official evaluation for a non-official scene list" >&2
  exit 1
fi

bash "$EVAL_RUNNER" "$EXPERIMENT_NAME" "$PRED_ROOT"
echo "[$(date '+%F %T')] Profile completed: $PROFILE"
