#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE_ROOT="${BOXFUSION_TR3D_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev}"
GPU_SPEC="${1:-0,1}"
RUN_TAG="${2:-tr3d_t1_epoch12_fp32_observer10_v1}"
CHECKPOINT="${BOXFUSION_TR3D_CHECKPOINT:-$SOURCE_ROOT/work_dirs/tr3d/tr3d_fg_full_seed0_fp32_v1/epoch_12.pth}"
CONFIG="${BOXFUSION_TR3D_CONFIG:-$ROOT/config/tr3d/tr3d_scannet_foreground_official_val.py}"
ENV_REF="${BOXFUSION_TR3D_ENV:-$SOURCE_ROOT/.conda/boxfusion-tr3d}"
INPUT_MANIFEST="${BOXFUSION_TR3D_INPUT_MANIFEST:-$SOURCE_ROOT/data/tr3d_scannet/scene_manifest.jsonl}"
SCENE_LIST="${BOXFUSION_TR3D_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
G0_MANIFEST="${BOXFUSION_G0_MANIFEST:-$ROOT/manifests/frozen_g0_selective_boxer_full100.json}"
B6_MANIFEST="${BOXFUSION_B6_MANIFEST:-$ROOT/manifests/frozen_b6_full100.json}"
ALLOW_BUSY="${BOXFUSION_TR3D_ALLOW_BUSY_GPUS:-0}"
RUN_SMOKE="${BOXFUSION_TR3D_RUN_SMOKE:-1}"
PREFLIGHT_ONLY="${BOXFUSION_TR3D_PREFLIGHT_ONLY:-0}"
EXPECTED_CHECKPOINT_SHA256="${BOXFUSION_TR3D_EXPECTED_CHECKPOINT_SHA256:-a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448}"

for path in "$CHECKPOINT" "$CONFIG" "$INPUT_MANIFEST" "$SCENE_LIST" "$G0_MANIFEST" "$B6_MANIFEST"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
[[ -x "$ENV_REF/bin/python" ]] || { echo "Invalid TR3D environment: $ENV_REF" >&2; exit 2; }
[[ -d "$ROOT/third_party/mmdetection3d" ]] || { echo "Missing pinned MMDetection3D vendor tree" >&2; exit 2; }
[[ "$ALLOW_BUSY" == "0" || "$ALLOW_BUSY" == "1" ]] || { echo "BOXFUSION_TR3D_ALLOW_BUSY_GPUS must be 0/1" >&2; exit 2; }
[[ "$RUN_SMOKE" == "0" || "$RUN_SMOKE" == "1" ]] || { echo "BOXFUSION_TR3D_RUN_SMOKE must be 0/1" >&2; exit 2; }
[[ "$PREFLIGHT_ONLY" == "0" || "$PREFLIGHT_ONLY" == "1" ]] || { echo "BOXFUSION_TR3D_PREFLIGHT_ONLY must be 0/1" >&2; exit 2; }

checkpoint_sha256="$(sha256sum "$CHECKPOINT" | awk '{print $1}')"
[[ "$checkpoint_sha256" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
  echo "Checkpoint SHA256 mismatch: $checkpoint_sha256" >&2
  exit 2
}
python "$ROOT/tools/verify_frozen_anchor_manifest.py" --manifest "$G0_MANIFEST"
python "$ROOT/tools/verify_frozen_anchor_manifest.py" --manifest "$B6_MANIFEST"
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "Preflight OK: checkpoint/config/environment/vendor/B6/G0 anchors"
  exit 0
fi

gpu_compute_pids() {
  local gpu="$1"
  local output
  if ! output="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>&1)"; then
    echo "nvidia-smi query failed for GPU $gpu: $output" >&2
    return 2
  fi
  printf '%s\n' "$output" | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print}'
}

if [[ "$ALLOW_BUSY" != "1" ]]; then
  IFS=',' read -r -a requested_gpus <<< "$GPU_SPEC"
  for gpu in "${requested_gpus[@]}"; do
    if ! busy="$(gpu_compute_pids "$gpu")"; then
      echo "Refusing launch because GPU occupancy could not be verified" >&2
      exit 3
    fi
    if [[ -n "$busy" ]]; then
      echo "Refusing to compete with active jobs on GPU $gpu (PIDs: $(echo "$busy" | tr '\n' ' '))" >&2
      exit 3
    fi
  done
fi

RUNTIME_ROOT="$ROOT/.runtime/$RUN_TAG"
mkdir -p "$RUNTIME_ROOT"/{tmp,xdg,torch,matplotlib,cuda}
export TMPDIR="$RUNTIME_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_ROOT/xdg"
export TORCH_HOME="$RUNTIME_ROOT/torch"
export MPLCONFIGDIR="$RUNTIME_ROOT/matplotlib"
export CUDA_CACHE_PATH="$RUNTIME_ROOT/cuda"
export PYTHONNOUSERSITE=1
export PATH="$ENV_REF/bin:$PATH"

echo "Isolated trained-TR3D observer experiment"
echo "  root: $ROOT"
echo "  checkpoint: $CHECKPOINT"
echo "  checkpoint sha256: $checkpoint_sha256"
echo "  GPUs: $GPU_SPEC"
echo "  full-scene ceiling only: prefix_id=full"
echo "  BoxFusion score threshold remains untouched (G0 metadata: 0.4)"

if [[ "$RUN_SMOKE" == "1" ]]; then
  first_gpu="${GPU_SPEC%%,*}"
  smoke_tag="${RUN_TAG}_smoke1"
  BOXFUSION_TR3D_ENV="$ENV_REF" \
  BOXFUSION_TR3D_CONFIG="$CONFIG" \
  BOXFUSION_TR3D_CHECKPOINT="$CHECKPOINT" \
  BOXFUSION_TR3D_INPUT_MANIFEST="$INPUT_MANIFEST" \
  BOXFUSION_TR3D_PREFIX_ID=full \
  BOXFUSION_TR3D_SCORE_THRESHOLD=0.01 \
  BOXFUSION_TR3D_MAX_PROPOSALS=1000 \
    bash "$ROOT/scripts/run_tr3d_single_scene_smoke.sh" \
      "$first_gpu" "$smoke_tag"
fi

COMMON_ENV=(
  BOXFUSION_TR3D_ENV="$ENV_REF"
  BOXFUSION_TR3D_CONFIG="$CONFIG"
  BOXFUSION_TR3D_CHECKPOINT="$CHECKPOINT"
  BOXFUSION_TR3D_INPUT_MANIFEST="$INPUT_MANIFEST"
  BOXFUSION_TR3D_SCENE_LIST="$SCENE_LIST"
  BOXFUSION_TR3D_PREFIX_ID=full
  BOXFUSION_TR3D_SCORE_THRESHOLD=0.01
  BOXFUSION_TR3D_MAX_PROPOSALS=1000
  BOXFUSION_TR3D_CACHE_ROOT="$ROOT/cache/tr3d_residual/$RUN_TAG"
  BOXFUSION_TR3D_LOG_ROOT="$ROOT/logs/tr3d/$RUN_TAG"
  BOXFUSION_TR3D_FROZEN_MANIFEST="$G0_MANIFEST"
)

env "${COMMON_ENV[@]}" \
  bash "$ROOT/scripts/run_tr3d_observer10.sh" "$GPU_SPEC" "$RUN_TAG"

for anchor in b6 g0; do
  if [[ "$anchor" == "b6" ]]; then
    manifest="$B6_MANIFEST"
  else
    manifest="$G0_MANIFEST"
  fi
  report_root="$ROOT/reports/tr3d/$RUN_TAG/$anchor"
  env "${COMMON_ENV[@]}" \
    BOXFUSION_TR3D_FROZEN_MANIFEST="$manifest" \
    BOXFUSION_TR3D_REPORT_ROOT="$report_root" \
    bash "$ROOT/scripts/audit_tr3d_observer.sh" "$RUN_TAG"
  python "$ROOT/tools/summarize_tr3d_observer.py" \
    "$report_root/union_oracle.json" | tee "$report_root/summary.txt"
done

echo "Experiment complete: $ROOT/reports/tr3d/$RUN_TAG"
